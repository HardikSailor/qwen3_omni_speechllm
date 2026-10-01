# Qwen3-Omni training pipeline: plan and code provenance

> **Cluster note (2026-09-30):** this document was written on the previous cluster (Orion: Slurm, H200, Apptainer). The project now runs on NSCC (PBS, H100, enroot). Map the old paths with the *Paths* section of `README.md`; the container is described in `container/README.md`. Job numbers (e.g. 150431) are Orion Slurm jobs.

Written 2026-09-28 as a plan; **status 2026-09-30: `omni_mds/` and `train_omni.py` are built and tested (progress and results in §10); the eval script does not exist yet.** Both trainers run on NSCC (H100, PBS, enroot).

This document describes the pipeline we will build to fine-tune Qwen3-Omni-30B-A3B on our SEA/SG MDS data. For each part it says where the code comes from:
- **[OLD]** the old MERaLiON trainer, `toolkits/multimodal_trainer`;
- **[SWIFT]** ms-swift / Megatron-SWIFT, `toolkits/ms-swift` @ 8ec0455, the same commit as inside the container;
- **[NEW]** code we write.

Related documents:
- `MDS_DATA_PIPELINE.md`: the data-side design and evidence (§0 overview, §0.5 streaming vs. map-style, §4 design details);
- `FINETUNING_GUIDE.md`: training stages, tasks and languages (§12 on the training stack);
- `WORKLOG.md`: daily progress;
- `DECISION_LOG.md`: why each decision was made, with evidence, problems found, jobs and file index;
- `/scratch/prj0000000234/sailorhb/container/README.md`: the container and `run_container.sh`.

---

## 1. Principles

1. **ms-swift stays unmodified.** All our changes live in our own package and entry script, which subclass ms-swift classes and override one or two methods. Upgrading ms-swift then means re-checking a few method signatures, not re-applying patches.
2. **Don't write our own training loop.** ms-swift and Megatron-SWIFT already own the parts that are hard and already tested here: model loading, LoRA, expert parallelism, checkpointing and resume, PEFT export, logging (see `MDS_DATA_PIPELINE.md` §0.5).
3. **Reuse the old trainer's data knowledge, not its model code.** The text normalisation, multilingual prompts, dataset mixes and metrics encode years of SEA-specific decisions. The collators, trainer, FSDP code and model classes are tied to the MERaLiON encoder+adapter+LLM design and are dropped.
4. **Copy (vendor) the old pieces instead of importing them.** `multimodal_trainer/modules` pulls in mosaic, torch.distributed and model code at import time. Copy the pure-Python files we need into our package, with a header comment naming the source file, and test them against the originals.
5. **One data layer, two trainers.** The same dataset object feeds `swift sft` (HF trainer, for prototyping) and `megatron sft` (for scale).
6. **Runs on H100 (80 GB) and H200 (141 GB).** No hardware constants in code; per-cluster profiles set memory-related settings; every smoke test also runs with an 80 GB memory cap (§9).

## 2. The pipeline at a glance

```
 ┌────────────────────────────── DATA ──────────────────────────────┐
 │ MDS shards (.mds.zstd)                     [existing data, zoux]  │
 │ /scratch/prj0000000234/zoux/datasets/datasets_mosaic_stage_...   │
 │ mix YAML: name, path, task, choose, repeat, weight   [OLD format] │
 │        │                                                          │
 │        ▼                                                          │
 │ READER (choose one, see §3.1)                                     │
 │   D: unzip once + index + pread           [NEW]                   │
 │   C: mosaic StreamingDataset + Stream     [OLD init_train_stream] │
 │        │  raw sample: context_audio, instruction_text,            │
 │        │  answer_text, context_text, language, task               │
 │        ▼                                                          │
 │ row_fn (per sample)                         [PORTED from OLD]     │
 │   normalise answer, pick multilingual prompt, instruction dropout │
 │   optional audio augmentation               [ADAPTED from OLD]    │
 │        │  {"messages":[user "<audio>"+prompt, assistant answer],  │
 │        │   "audios":[raw bytes]}                                  │
 └────────┼──────────────────────────────────────────────────────────┘
          ▼
 ┌──────────────────────────── TRAINING ────────────────────────────┐
 │ Qwen3-Omni template: audio features, audio tokens (13/s),         │
 │   chat format, labels, padding-free collator         [SWIFT]      │
 │        ▼                                                          │
 │ swift sft (HF Trainer, DDP/DeepSpeed)                [SWIFT]      │
 │   or megatron sft (EP=8, mcore-bridge)               [SWIFT]      │
 │   LoRA on explicit target list, no audio-encoder LoRA [NEW list]  │
 │        ▼                                                          │
 │ checkpoints: PEFT adapter (HF format)                [SWIFT]      │
 └────────┼──────────────────────────────────────────────────────────┘
          ▼
 ┌─────────────────────────── EVALUATION ───────────────────────────┐
 │ MDS val sets → same reader + row_fn (no augmentation) [NEW/OLD]   │
 │ generation: HF+PEFT adapter, or merged model in vLLM  [SWIFT]     │
 │ normalisers + WER/MER, BLEU, METEOR, weighted score   [OLD]       │
 │ predictions JSONL + scores per checkpoint, best/ link [NEW]       │
 └───────────────────────────────────────────────────────────────────┘
```

## 3. Components and where their code comes from

### 3.1 Data

| Component | Source | Action | Notes |
|---|---|---|---|
| MDS shards | `zoux/.../datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal/{train,…}` | Use as is | 117 readable train datasets, 5,604 shards, 1.81 TB. Owned by `zoux`: confirm it is current before relying on it |
| Mix YAMLs | [OLD] `multimodal_trainer/config/dataset/*.yaml` (e.g. `meralion2_ctm_2611_with_new_data_with_sampling_v4.3.yaml`) | Reuse the format; write our own `mixes/*.yaml` | Drop or fix the unreadable `wonghmj/.../SQA` paths. Optional new per-dataset keys: `language`, `prompt_type`, `max_seconds` |
| Reader, option D (map-style) | [NEW] `omni_mds/reader.py`, `build_mds_index.py`, `stage_mds.py` | Write | Sketch in `MDS_DATA_PIPELINE.md` §5. Needs a one-time unzip (to node NVMe or once to `/scratch`) and a per-dataset parquet index |
| Reader, option C (mosaic streaming) | [OLD] `modules/datasets/datasets.py:8` `init_train_stream`, `:19` `init_train_streaming_dataset` | Copy and adapt | Remove the hard-coded `// 8` GPUs-per-node; add `StreamingDataLoader.state_dict()` save/load for resume |
| Mixing (`choose` / `repeat`) | Option D: [NEW] `omni_mds/dataset.py` `MDSMixDataset`. Option C: native mosaic `Stream(choose=, repeat=)` | Write / reuse | Same semantics: re-drawn every epoch from a seed |
| MDS bytes → audio | [OLD] `modules/data_collators/data_collator.py` `_convert_mds_*` | Not needed | ms-swift takes raw audio bytes directly (`swift/template/vision_utils.py:129` `load_file`) |
| Answer normalisation | [OLD] `modules/data_collators/instruct_collator.py:32` `_standardize_response` | Port into `omni_mds/sea_text.py` | Fillers, Chinese/Thai handling, gigaspeech2-id lower-casing, hashtags, apostrophes, speaker tags. **Decide the target convention first** (see §7) |
| Prompts | [OLD] `modules/data_collators/prompt_utils.py:164` `get_asr_prompt`, `:174` `get_ac_prompt`, `:73` `_get_prompt`; `instruct_collator.py:65` `get_prompt` | Port | Keep `random` / `random_multilingual` / `dataset` prompt types and the Vietnamese format-hint augmentation |
| Instruction library (13 languages: my, zh, en, id, ja, ko, lo, ms, ne, tl, ta, th, vi) | [OLD] `modules/data_collators/instructions_lib/*.py` | Copy as is | Pure Python |
| Instruction dropout | [OLD] `instruct_collator.py:124–131` | Port, or better make text-only instructions their own mix stream | A text-only row without `audios` is valid in ms-swift |
| Chat format / system prompt | [OLD] `instruct_collator.py:148` `get_llm_chat_template`, `<SpeechHere>` split | **Drop** | Replaced by Qwen format: user = `<audio>` + instruction, no MERaLiON system prompt |
| Audio augmentation | [OLD] `modules/data_collators/data_augmentation.py` (`add_noise`, `add_rir_effect`, `change_speed`, `change_pitch`, `_augment_sample`; `augment_data:268` is batch-based) | Adapt | Crowd noise was mixed from other clips in the batch; make it per-sample with a small per-worker noise reservoir. Re-encode to WAV bytes afterwards |
| Language IDs | [OLD] `modules/data_collators/language_mapping.py:24` `get_language_id` | Optional | Only if we add language tokens (guide §5.7) |
| `context_text` | MDS column | **Never in the prompt** | It is the transcript for ST and ER; it would leak the answer. Optional use as a chain-of-thought target |

### 3.2 Training

| Component | Source | Action | Notes |
|---|---|---|---|
| Entry script | [NEW] `train_omni.py {sft\|megatron} …` | Write (~50 lines) | Subclasses `SwiftSft` / `MegatronSft` and overrides `_get_dataset` (both modes). For option C also overrides the dataloader (below) |
| Dataset hook | [SWIFT] `swift/pipelines/train/sft.py:81` `SwiftSft._get_dataset`; `MegatronSft` (`swift/megatron/pipelines/train/sft.py:23`) inherits it | Override | Returns our train/val dataset objects |
| Lazy encoding | [SWIFT] `swift/dataset/utils.py:57` `LazyLLMDataset`, applied in `sft.py:128` `_post_process_datasets` | Use as is | Always `--lazy_tokenize true`: encodes inside dataloader workers, skips bad samples |
| Dataloader, option D | [SWIFT] HF: `BatchSamplerShard` in `swift/trainers/mixin.py:1398`; Megatron: `MegatronPretrainingRandomSampler` in `swift/megatron/trainers/base.py:1045` | Use as is | Per-rank sampling, `group_by_length`, exact resume |
| Dataloader, option C | [SWIFT] same two methods | Override both | Replace the rank-0 `DataLoaderDispatcher` with a plain per-rank DataLoader over the mosaic partition |
| Template | [SWIFT] `swift/template/templates/qwen.py:1208` `Qwen3OmniTemplate` (registered at `:1219`, `default_system=None`) | Use as is | Audio features, `<\|audio_pad\|>` × 13 tokens/s, labels, padding-free |
| Packing | [SWIFT] `PackingDataset` / `IterablePackingDataset` (`swift/pipelines/train/sft.py:142`) | Later | Option D + Megatron `--packing` needs a small subclass fed with index lengths. Start with `--padding_free true` |
| Model loading | [SWIFT] `swift/pipelines/utils.py:37` `prepare_model_template`; Megatron via mcore-bridge (loads HF safetensors directly) | Use as is | Weights: `/scratch/prj0000000234/sailorhb/hf_models/` |
| LoRA | [SWIFT] `swift/pipelines/train/tuner.py:370` `TunerMixin.prepare_model` | Use as is, with **our explicit target list** | `lora_targets_thinker_attn_audio_proj.txt` (194 modules) or `container/bench_8gpu/targets_attn_only.txt`. Never a regex: PEFT 0.20 leaks it into the audio encoder |
| HF trainer | [SWIFT] `swift/trainers/seq2seq_trainer.py:26` `Seq2SeqTrainer` | Use as is | For prototyping, data/prompt work and architecture changes |
| Megatron trainer | [SWIFT] `megatron sft`, EP=8, TP=PP=CP=1 | Use as is | For large runs, expert LoRA, full fine-tuning, multi-node. Needs `NPROC_PER_NODE`; log arg is `--logging_steps` |
| Environment | `/scratch/prj0000000234/sailorhb/container/swift_megatron_cu128.sif` via `run_container.sh` | Use; small edit | Forward `PYTHONPATH` and `SWIFT_AUDIO_LOAD_BACKEND` (add to `FORWARD=(…)`). nemo_env/`swift_omni` venv only for 1-GPU debugging |

Old training code that is **dropped**: `multimodal_trainer/train.py`, `modules/trainer.py`, `modules/distributed/` (FSDP1/FSDP2), `modules/models/`, `modules/data_collators/batch_processor.py` and the collators' tokenisation.

### 3.3 Evaluation

| Component | Source | Action | Notes |
|---|---|---|---|
| In-training validation | [SWIFT] `--val_dataset` through the same `_get_dataset` override | Use | Loss only, on small fixed subsets per task/language (`choose: 512`). No `predict_with_generate`: too slow for a 30B MoE |
| Checkpoint eval script | [NEW] `eval_omni.py` | Write | Reads MDS val sets with the same reader + `row_fn` (augmentation off), generates with transformers + PEFT adapter (frequent checks) or with vLLM on a temporarily merged copy (final numbers; vLLM 0.17.1 can't load Qwen3-Omni adapters, `DEPLOYMENT.md` §1–2), writes predictions JSONL and scores per checkpoint, keeps a `best/` link |
| Generation backend | [SWIFT] `swift infer --infer_backend vllm` (vLLM 0.17.1 in the container) | Use | Base model tested: 3/3 LibriSpeech clips exact. **No LoRA support for Qwen3-Omni in vLLM 0.17.1** (`DEPLOYMENT.md` §1); use `--infer_backend transformers --adapters` or a merged copy |
| Metrics | [OLD] `modules/metrics/wer.py:6` `compute_wer`, `bleu.py:5`, `meteor.py:5`, `metrics/__init__.py:8` `get_metric_fn` (metric chosen per dataset path) | Copy as is | So scores stay comparable with MERaLiON results |
| Normalisers for scoring | [OLD] `modules/text_normalizers/wer_text_normalizer/` (`normalizer.py`, `basic.py`, `whisper_english.py`, `english.json`) | Copy as is | Needs jiwer, regex, more_itertools (all in the container) |
| Weighted aggregate score | [OLD] YAML `weight` + old trainer's per-set evaluation (`modules/trainer.py` `_evaluate_one_dataset`) | Re-implement in `eval_omni.py` | Keep the same weighting so numbers are comparable |
| Benchmarks | [OLD] `eval_dynamic_superb.py`, AudioBench scripts under `scripts/a2ap_scripts/` | Later | Need a Qwen3-Omni generation backend instead of the MERaLiON model |

## 4. Planned layout

All new code goes under `toolkits/qwen3_omni_speechllm/`. ms-swift and multimodal_trainer are not modified.

```
qwen3_omni_speechllm/
├── omni_mds/                     # [NEW] our package (on PYTHONPATH inside the container)
│   ├── mosaic_stream.py          #   reader (option C, chosen): mosaic Streams + per-rank dataloader
│   ├── mds_io.py                 #   direct MDS decoding (tests, peeking)
│   ├── dataset.py                #   MDSMixDataset: mix YAML → choose/repeat → row_fn
│   ├── sea_text.py               #   row_fn: ported normalisation, prompts, dropout
│   ├── augment.py                #   per-sample audio augmentation (adapted)
│   ├── instructions_lib/         #   copied from OLD, unchanged
│   ├── text_normalizers/         #   copied from OLD, unchanged
│   └── metrics/                  #   copied from OLD (wer, bleu, meteor)
├── tools/
│   └── vendored/                 # old MDS helper scripts (convert_hf_to_mds, generate_dataset_config, …)
├── mixes/
│   ├── sea_v1_train.yaml         # [NEW] our mix, same format as the old YAMLs
│   └── sea_v1_val.yaml
├── train_omni.py                 # [NEW] entry: python train_omni.py {sft|megatron} <ms-swift args>
├── eval_omni.py                  # [NEW] checkpoint eval (HF+PEFT or merged+vLLM) with the old metrics
├── tests/
│   ├── test_reader.py            #   reader output == reference decode, sample for sample
│   └── test_sea_text.py          #   row_fn targets == old InstructDataCollator on the same rows
├── profiles/                     # [NEW] per-cluster settings (§9.3), read by the job scripts
│   ├── h200_orion.env            #   141 GB, Slurm
│   └── h100_a2ap.env             #   80 GB, PBS (to confirm)
└── slurm/                        # job scripts (PBS equivalents for the H100 cluster)
    ├── smoke_1gpu.sbatch
    ├── train_8gpu.sbatch
    └── eval.sbatch
```

## 5. One sample, end to end

1. The sampler picks index `i`. Option D: the HF or Megatron sampler over the mix list. Option C: mosaic's own per-rank partition.
2. The reader returns the raw MDS sample: `context_audio` (WAV or Opus bytes), `instruction_text`, `answer_text`, `context_text`, `language`, `task`.
3. `row_fn`:
   - normalises `answer_text` for its task and language;
   - picks the prompt: the dataset instruction, or a random one in English or the native language;
   - may apply instruction dropout and audio augmentation;
   - returns `{"messages": [{"role": "user", "content": "<audio>" + prompt}, {"role": "assistant", "content": answer}], "audios": [audio_bytes]}`.
4. `LazyLLMDataset` calls `Qwen3OmniTemplate.encode` in the dataloader worker:
   - decodes the audio to 16 kHz;
   - computes Whisper-style features;
   - expands `<audio>` to 13 tokens per second;
   - tokenises the text and masks the prompt in the labels.
5. The template's data collator builds the batch, padding-free if enabled.
6. The ms-swift trainer runs the forward and backward pass (HF DDP or Megatron EP=8), with LoRA on the explicit targets only.
7. Checkpoints are saved as PEFT adapters, which `eval_omni.py` scores with transformers + PEFT, or with vLLM after a temporary merge (`DEPLOYMENT.md` §2.2).

## 6. Build phases

| # | Step | Output | Check |
|---|---|---|---|
| 0 | **Decide the reader** — **done 2026-09-28: C, mosaic streaming plug-in** (user decision) | Decision in this doc | `MDS_DATA_PIPELINE.md` §0.5 |
| 1 | Copy `instructions_lib`, normalisers and metrics; port `row_fn` into `sea_text.py` | `omni_mds/` text parts | `test_sea_text.py`: identical targets to the old `_standardize_response` on ~50 rows per task |
| 2 | Reader: `omni_mds/mosaic_stream.py` (option C) — **done 2026-09-28, see §10.2** | Reader + dataloader | Raw samples identical to direct decoding; `choose` per epoch; disjoint equal rank partitions; exact mid-epoch resume |
| 3 | `dataset.py` + `train_omni.py`; 1-GPU `swift sft` LoRA smoke test on a 3-dataset mix | Loss curve | `template.print_inputs` shows audio expanded at 13 tokens/s; no LoRA in `audio_tower`; also passes with `OMNI_GPU_MEM_GB=80` (§9.3) |
| 4 | 8-GPU run in the container (D: with staging) | samples/s, dataloader wait | Close to the JSONL benchmark (17 samples/s, job 147727); kill and resume gives the same next batch |
| 5 | `eval_omni.py` on the MDS val sets | Per-task/language WER/MER/BLEU | Zero-shot baseline of the base model (guide §1) |
| 6 | Full mix YAML, data inventory (hours per task × language) | `mixes/sea_v1_*.yaml`, hours table | Missing SQA paths fixed or removed |
| 7 | `train_omni.py megatron` | Megatron smoke test with EP=8 | Same checks as step 3; adapter exports cleanly |

Steps 1–3 are about 1–2 days. After step 3, switching between `swift sft` and `megatron sft` does not touch the data code.

## 7. Open decisions

| Decision | Options | Who / input |
|---|---|---|
| ~~Reader~~ | **Decided 2026-09-28: C (mosaic streaming plug-in)** | — |
| Data location | Is `zoux/.../datasets_mosaic_stage_AudioLLM_v2.1` the current, stable copy? | `zoux` / MERaLiON data owners |
| Missing SQA sets | Get access to `wonghmj/.../SQA/mds_qaed_cleaned_reasoning`, or drop them | `wonghmj` |
| Target text convention | **ASR normalisation decided 2026-09-28: off** (`RowConfig.asr_normalize_text=False`; set True to restore the old behaviour). Still open: keep or strip `<SpeakerN>:` tags and `#entity#` hashtags (also present in SDS targets) | Team. The model copies target style (LibriSpeech all-caps targets made WER worse) |
| System prompt | None (Qwen default) vs. one fixed short prompt, same in training and eval | Team |
| H100 cluster details | Scheduler, container runtime, data access, local disk (§9.4) | Team / cluster admins |
| Trainer for LoRA runs | `swift sft` (17.0 samples/s at 8 GPUs) vs. Megatron EP=8 (13.5 samples/s, but 14 GiB vs. 62 GiB per GPU, so larger batches or packing may win) | Megatron re-run with bigger batches / packing |

---

## 8. Parallelism: DDP, FSDP or expert parallelism

Added 2026-09-28. **Short answer:** DDP or Megatron expert parallelism (EP), depending on what we train. FSDP1 is out; FSDP2 / DeepSpeed ZeRO-3 is only a fallback.

### 8.1 What our ms-swift checkout offers

| Method | Available | Notes |
|---|---|---|
| DDP (plain data parallel) | Yes, default in `swift sft` | Used in benchmark 147727 |
| DeepSpeed ZeRO-0/1/2/3 (+offload) | Yes (`swift/config/zero*.json`) | |
| FSDP2 | Yes, the only FSDP preset (`--fsdp fsdp2`, `swift/config/fsdp2.json`; example `examples/train/multi-gpu/fsdp2_lora`) | Cannot be combined with DeepSpeed (`swift/arguments/sft_args.py:285`) |
| FSDP1 | No preset | Legacy. The old trainer used it for dense LLMs (Gemma, Llama); no reason to use it here |
| Megatron: EP / TP / PP / CP + distributed optimizer | Yes, in the container (`megatron sft`) | |

### 8.2 Model size (counted from the safetensors headers, 2026-09-28)

| Part | Parameters | bf16 |
|---|---|---|
| Thinker experts (48 layers × 128 experts) | 28.99 B | 58.0 GB |
| Thinker non-expert (attention, router, norms, embeddings, lm_head) | 1.54 B | 3.1 GB |
| Audio encoder (AuT) | 0.65 B | 1.3 GB |
| Vision encoder | 0.54 B | 1.1 GB |
| **Thinker total (what training loads)** | **31.72 B** | **63.4 GB = 59.1 GiB** |
| Talker + code2wav (speech output; not loaded for training) | 3.54 B | 7.1 GB |

Experts are 91% of the thinker. That is why splitting experts (EP) matters much more than splitting anything else.

### 8.3 By kind of training

| Training | Method | Why |
|---|---|---|
| LoRA on attention + audio projection (first runs) | **DDP** (`swift sft`) or **Megatron EP=8** | Frozen weights (59 GiB) fit on one GPU. Job 147727 at 8×H200: DDP 17.0 samples/s at 61.8 GiB/GPU; EP=8 13.5 samples/s at 14.2 GiB/GPU. EP's spare memory can go into bigger batches, packing or longer audio: re-run before choosing |
| LoRA on experts (`all-linear`) | **Megatron EP** | With DDP every GPU computes all 128 experts for its tokens. EP splits them and uses grouped GEMM |
| Full fine-tuning of the thinker or of the experts | **Megatron EP=8–16 + distributed optimizer** | Weights + grads + Adam ≈ 490 GB must be split. ms-swift's own Qwen3-30B-A3B full-FT benchmark (16×A800): Megatron 9.6 s/it vs. ZeRO-3 91.2 s/it; ZeRO-2 out of memory |
| HF-side prototype that doesn't fit with DDP (e.g. unfreezing the audio encoder + part of the thinker) | FSDP2 or ZeRO-3, as a fallback | They split parameters but not experts: every layer gathers all 128 experts' weights onto every GPU. Lots of communication; use only until the change is ported to Megatron |

### 8.4 Megatron layout

- **Within a node (8 GPUs):** experts split, EP=8. TP=PP=CP=1: hidden size 2048 is small and there are only 4 KV heads, so tensor parallelism doesn't pay off.
- **Across nodes:** data parallel. With full or expert training, the distributed optimizer splits optimizer state across data-parallel ranks.
- **Encoders:** audio (0.65 B) and vision (0.54 B) are copied onto every GPU.
- With TP=PP=CP=1 every GPU is its own data-parallel rank. That is the case the mosaic streaming option handles without changes (`MDS_DATA_PIPELINE.md` §0.5).

**Plan:** DDP `swift sft` for LoRA prototyping → re-run the Megatron benchmark with bigger batches or packing → Megatron EP for large/multi-node runs, expert LoRA and full fine-tuning.

---

## 9. Running on H100 (80 GB) as well as H200 (141 GB)

Added 2026-09-28. Requirement: the same code and pipeline must run on 80 GB H100s (e.g. the team's ASPIRE2A+ machines, where the old trainer ran) as well as on our 141 GB H200s.

### 9.1 What carries over unchanged

H100 and H200 are the same Hopper architecture (sm_90). The container's CUDA 12.8 build, flash-attn, Transformer Engine, grouped GEMM and vLLM kernels are the same. **The only differences that matter are memory (80 vs. 141 GB) and memory bandwidth.** So portability is about memory settings and job scripts, not code paths.

### 9.2 Memory per GPU: H200 vs. H100

Measured values are from job 147727 (8×H200, micro-batch 1, gradient checkpointing, short LibriSpeech clips). Everything marked "est." is a calculation from §8.2 and still needs a test.

| Training | H200 141 GB | H100 80 GB (~79 GiB usable) | Plan on H100 |
|---|---|---|---|
| LoRA attention, **DDP** | 61.8 GiB (measured) | Fits, but only ~17 GiB left for activations | Allowed with guardrails (§9.3): micro-batch 1, gradient checkpointing on, `max_length` ≤ 2048, audio ≤ 30 s. Longer answers or bigger batches can run out of memory |
| LoRA attention, **Megatron EP=8** | 14.2 GiB (measured) | Same, comfortable | **Preferred default on H100** |
| LoRA on experts, Megatron EP=8 | est. ~15–20 GiB | Same | Fine |
| Full fine-tuning, Megatron EP=8, 1 node | est. ~80 GB + activations | Does not fit | Needs ≥2 nodes (below) |
| Full fine-tuning, Megatron EP=8 × DP=2 (2 nodes) | est. ~56 GB + activations | est. fits, tight | Or EP=16, or Megatron optimizer CPU offload |
| vLLM eval (thinker, bf16 63 GB) | TP=1 | TP=1 leaves only ~10 GB for KV cache | Use `tensor_parallel_size=2` on H100 |

How the full fine-tuning estimate is made (Megatron, EP=8, audio and vision encoders frozen):
- Per GPU: experts 29.0/8 = 3.62 B, plus 1.54 B non-expert (trainable), plus 1.19 B encoders (frozen) = 6.35 B parameters.
- bf16 weights 12.7 GB; fp32 gradients for the 5.16 B trainable 20.6 GB.
- Optimizer state (fp32 master + Adam m, v = 12 bytes/param), split across data-parallel ranks. Expert state is split only across the expert-data-parallel group (= number of nodes when EP=8), so: 1 node ≈ 45.7 GB, 2 nodes ≈ 22.9 GB, 4 nodes ≈ 11.5 GB.
- Total before activations: 1 node ≈ 79 GB, 2 nodes ≈ 56 GB, 4 nodes ≈ 45 GB.

### 9.3 Design rules so the code runs on both

1. **No hardware constants in code.**
   - Never hard-code GPUs per node. The old `init_train_stream` has `dist.get_rank() // 8`, which must go.
   - Take GPUs per node, node count, partition and memory from the job environment or a profile file.
2. **Hardware profiles, not separate scripts.** One file per cluster, e.g. `profiles/h200_orion.env` and `profiles/h100_a2ap.env`, setting:
   - GPU memory class;
   - GPUs per node;
   - micro-batch and `max_length`;
   - EP size;
   - vLLM tensor-parallel size;
   - cache / staging directory;
   - scheduler (Slurm here; PBS on ASPIRE2A+).

   The launch scripts read the profile; the Python code only sees ordinary ms-swift arguments.
3. **Memory settings fixed by the profile, not tuned by hand per run:**
   - gradient checkpointing on;
   - `--max_length`;
   - micro-batch, with gradient accumulation to keep the global batch the same across clusters;
   - `--padding_free true`;
   - packing length.

   Same global batch and learning-rate schedule on both clusters, so results are comparable.
4. **Filter long samples before encoding.** The index (option D) or a duration check in `row_fn` (option C) drops samples whose estimated tokens (13/s audio + text) exceed `max_length`. On an 80 GB GPU this prevents rare out-of-memory crashes from one long sample.
5. **Test the H100 budget on our H200s.** The entry script gets an optional `OMNI_GPU_MEM_GB=80` setting. When set, it calls `torch.cuda.set_per_process_memory_fraction(80/141)` before loading the model. Every smoke test runs once in this mode, so a setting that won't fit on an H100 fails here first.
6. **Container portability.** We use an Apptainer `.sif`; the old trainer used enroot `.sqsh` on ASPIRE2A+. Check whether ASPIRE2A+ offers Apptainer/Singularity. If not, convert the image with enroot or build from the same `.def`. The job scripts must support both Slurm and PBS; keep the Python entry point identical.
7. **Check per-node resources on the H100 cluster:** local NVMe size (for the unzip staging or the mosaic cache), CPU RAM, and access to the MDS data (the data lives on our `/scratch`; ASPIRE2A+ used `/data/projects/13003558`).

### 9.4 Open questions for H100

| Question | Why it matters |
|---|---|
| ~~Which H100 cluster exactly (ASPIRE2A+?), GPUs per node, interconnect~~ **Answered:** NSCC A2AP, 8 x H100 per node, InfiniBand (308-416 GB/s all_reduce busbw across 2-4 nodes, §10.12) | Sets EP size and whether multi-node Megatron is practical |
| Apptainer/Singularity available there? | Otherwise the image must be converted to enroot |
| Is the MDS data reachable from there, or does it need a copy? | Decides the reader and staging plan on that cluster |
| Node-local disk size | Unzip staging (~2 TB) or mosaic cache (600 GB) |

---

## 10. Progress

### 10.1 Phase 1 done (2026-09-28): row_fn port for all tasks, ST pilot mix

**Built** (all under `qwen3_omni_speechllm/`):

| File | What |
|---|---|
| `omni_mds/sea_text.py` | `build_row(sample, meta, cfg, rng, is_train, with_meta, augment)`: one raw MDS sample → one ms-swift row. Ported ASR normalisation, ASR/AC random (multilingual) prompts with the Vietnamese format hints, instruction dropout; language resolution (MDS column → mix YAML → dataset name) |
| `omni_mds/mix.py` | `load_mix()`: old YAML format plus optional `language` / `src_lang` / `tgt_lang`; checks keys, duplicate names and paths |
| `omni_mds/mds_io.py` | MDS v2 sample decoding, and streaming reads of the first N samples of a `.mds.zstd` shard (for tests and peeking; not the training reader) |
| `omni_mds/audio/augment.py` | Per-sample wrapper around the old batch augmentation: crowd noise from a per-worker reservoir of recent clips, same probabilities as the old `augment_data`, output re-encoded as WAV PCM16 16 kHz |
| `omni_mds/{instructions_lib, text_normalizers, metrics, audio/data_augmentation.py, language_mapping.py, prompt_utils_orig.py}`, `tools/vendored/` | Copied unchanged from the old trainer for **all** tasks, not only ST; provenance and md5s in `omni_mds/VENDORED.md` |
| `mixes/st_v0.yaml` | ST-only pilot mix: all 8 ST train sets (~720k samples/epoch via `choose`), 6 CoVoST2 validation sets (`choose: 512`) |
| `tests/test_sea_text.py` | Parity and format tests (run in the container, CPU, ~3 min) |

**Test results** (real MDS rows: first 50 of shard 0 from 32 dataset splits):
- **Prompt and target identical to the old `InstructDataCollator`** for the same row and seed: ST 2,792/2,792, ASR 2,000/2,000 (10 datasets covering en, codeswitch, id, th, vi, zh, ms, ta, yue), PQA/SDS/CPQA/AQA/AC 1,600/1,600, each under 4 configs (prompt type random / random_multilingual / dataset, normalisation on/off, Vietnamese prompt augmentation).
- Instruction dropout picks the same replacement as the old code for the same seed; augmented audio decodes as valid ≤30 s 16 kHz WAV; `context_text` never appears in the prompt; no dropout or augmentation at eval time.

**Deliberate differences from the old code** (each covered by a test):
- per-sample `random.Random` instead of the global `random`, so rows are reproducible per sample;
- AC prompts follow `ac_prompt_type` (old code used `asr_prompt_type`);
- `asr_normalize_text` defaults to **False** (old default true: lower-cased, unpunctuated ASR targets). This is still an open decision (§7);
- no MERaLiON system prompt and no `<SpeechHere>` template; the user turn is `<audio>` + instruction.

Kept as in the old code: instruction dropout **keeps the audio** (`dropout_keep_audio=True`; set False for text-only rows). For clips under 500 samples, the old augmentation replaces the target with `"EMPTY"`.

**Environment fixes:**
- The container has jiwer 4.x, which removed `compute_measures`. The copied `metrics/wer.py` falls back to `jiwer.process_words` (the only local patch to a copied file).
- METEOR downloads NLTK data at run time. It is now pre-downloaded to `/scratch/prj0000000234/sailorhb/container/cache/nltk_data`.
- `run_container.sh` now sets `NLTK_DATA` to that folder and forwards `PYTHONPATH`, `SWIFT_AUDIO_LOAD_BACKEND` and `OMNI_GPU_MEM_GB`. Backup of the previous version: session scratchpad.

**Data findings:**
- The MDS `language` column is empty in all ST, PQA, SDS and AQA rows seen, and missing (`None`) in CPQA. So language comes from the mix YAML or the dataset name.
- The ST "validation" sets are the CoVoST2 **test** split.
- CoVoST2 ta→en targets are loose and unpunctuated. People's Speech en→ms targets look machine-translated, with numbers spelled out.
- Non-ASR targets are never normalised, so SDS targets still contain `#entity#` hashtags (e.g. `#char-kway-teow#`). Include this in the target-convention decision.
- The instruction library covers 13 languages (earlier docs said 14).

**Next: phase 2 (reader)**, which needs the phase 0 decision (map-style vs. mosaic streaming; unzip job 149550 still waiting in the queue).

### 10.2 Phase 2 done (2026-09-28): mosaic streaming reader (option C)

**Decisions (user, 2026-09-28):** reader = **option C, mosaic streaming plugged into ms-swift**; **ASR normalisation off** by default (kept as the `asr_normalize_text` option).

**Built:**

| File | What |
|---|---|
| `omni_mds/mosaic_stream.py` | `OmniStreamingDataset` (subclass of mosaic `StreamingDataset`): one `Stream` per mix entry (`remote` = dataset folder on /scratch, read only; `local` = `<cache_root>/<host>/<split>/<name>`, node-local). `get_item` runs `build_row` (+ optional `transform`, e.g. template encode) with a per-sample RNG seeded by (seed, epoch, sample id). `StreamConfig` (defaults = old trainer: py1e shuffle, block 1.5 M, cache limit 600 GB; `for_validation()` = no shuffle, `sampling_method='fixed'`). `OmniStreamingDataLoader` + `make_dataloader`. `clean_stale_shared_memory` |
| `toolkits/pydeps/` | mosaicml-streaming 0.13.0 + zstd, python-snappy, cramjam, catalogue, installed with `pip install --no-deps --target` from inside the container. Its declared deps (transformers<5, torchvision, cloud SDKs) are **not** installed; the container's numpy 1.26.4 / torch 2.10 / transformers 5.2 stay in use (checked). Add it to `PYTHONPATH` |
| `tests/test_mosaic_stream.py` | 8 tests on the three small CoVoST2 X→en sets (CPU, login node, ~15 s) |
| `run_container.sh` | also forwards `OMNI_MDS_CACHE` |

**Test results (all pass):**
- raw samples from mosaic are identical to direct decoding of the same shard (`mds_io`); each sample maps to the right stream / mix entry;
- rows equal `build_row(raw, meta, cfg, Random(seed-epoch-id))`;
- `choose` gives exactly the requested count per dataset per epoch, without replacement, re-drawn each epoch (`balanced`); validation (`fixed`) gives the same subset in the same order each epoch;
- 2 ranks on one node (shared cache, as with 8 GPUs per node): disjoint partitions of equal size covering the whole epoch;
- mid-epoch resume via `state_dict()` / `load_state_dict()` reproduces the next batches exactly;
- unusable samples (no audio, decode/encode error) are **replaced** by a neighbour, never dropped, so every rank yields the same number of batches (dropping would hang DDP);
- throughput: ~3,400 rows/s in one process (first epoch, including shard copy + unzip; `build_row` only, no encoding). Training needs ~2–3 samples/s per GPU, so row building is not a bottleneck; template encoding (audio features) will dominate and is measured in phase 3.

**Problems found and fixed:**
- mosaic's `StreamingDataLoader` counts samples as `len(first value of the collated batch)`. With ms-swift's padding-free collator `input_ids` is `[1, total_tokens]`, so it would count 1 per batch and **resume from the wrong position**. `OmniStreamingDataLoader` counts `batch_size` per batch (exact: `drop_last=True` and no dropped samples). `make_dataloader` also checks that the dataset and dataloader batch sizes match (mosaic partitions by the dataset's batch size).
- mosaic starts `torch.distributed` itself when `WORLD_SIZE > 1`; torchrun provides `MASTER_ADDR`/`PORT` in real runs.
- `_meta.sample_id` is stored as a plain int so eval can write it to JSONL.

**Cautions:**
- `clean_stale_shared_memory()` removes every mosaic shared-memory segment this user owns on the node, including another running job's. Only call it when the job has the node to itself.
- Two simulated *nodes* cannot run on one host (mosaic keys shared memory per host). The multi-node partition check needs the 2-node run (phase 4). *(Done 2026-10-01: disjoint on 2 and 4 nodes, §10.12.)*
- Partitioning uses the global rank (`RANK`, `WORLD_SIZE`). Correct for DDP and Megatron EP with TP=PP=CP=1; with TP/CP > 1, set these to the data-parallel rank/size.
- Cache size per node: the ST pilot touches up to ~200 GB of shards (GigaSpeech 107 GB + People's Speech 73 GB + CoVoST2); the full mix ~2 TB, so the 600 GB `cache_limit` evicts and re-unzips each epoch. Set per cluster in the profile (ASPIRE2A+ local disk size is still unknown).

**Corrections found on the way:**
- Sample counts: count each dataset's **top-level** `index.json` only; it already covers the `0/ … 15/` sub-folders. My first ST counts summed both and were 2× too high (fixed in `mixes/st_v0.yaml`, ~610 k samples/epoch). The full-mix total (81.9 M) was correct.
- Missing paths in the v4.3 mix: **78** of 195 train (71 wonghmj SQA + 7 wonghmj AC) and 20 of 68 validation, not 98 as written earlier (fixed in `MDS_DATA_PIPELINE.md`).

**Next: phase 3**: `train_omni.py` (ms-swift entry: `_get_dataset` + dataloader overrides for HF and Megatron, template encode as `transform`, dataloader state in checkpoints) and a 1-GPU LoRA smoke test on `mixes/st_v0.yaml`, also with `OMNI_GPU_MEM_GB=80`.

### 10.3 Phase 3 in progress (2026-09-28): entry script and 1-GPU smoke test

**Built:**

| File | What |
|---|---|
| `train_omni.py` | `python train_omni.py sft --mix <yaml> [omni options] <swift sft args>`. `OmniSwiftSft(SwiftSft)` overrides `_prepare_dataset` → `OmniStreamingDataset` train/validation with `transform = template.encode`. `TrainerFactory.get_trainer_cls` is wrapped so the trainer: builds per-rank mosaic dataloaders (`get_train_dataloader` / `get_eval_dataloader`), saves `omni_mosaic_state.json` in every checkpoint, and on `--resume_from_checkpoint` loads it and sets `ignore_data_skip` (HF's skipping would re-read and re-encode every skipped sample). `OMNI_GPU_MEM_GB` caps GPU memory (H100 emulation). ms-swift requires `--dataset` and needs `--val_dataset` to keep evaluation on, so the script passes placeholders (`OMNI_MDS_MIX_TRAIN` / `_VALIDATION`) that are never loaded; a real `--dataset` is refused. `megatron` mode: not yet (phase 7) |
| `tests/check_encode_cpu.py` | CPU check of the whole data path without the model (mosaic → build_row → template encode → padding-free collator) |
| `slurm/smoke_1gpu.sbatch`, `tests/summarize_smoke.py` | 1-GPU smoke test: run A (40 steps, checkpoint + eval at 20/40), B (resume from A's step 20; losses 21–40 must match A), C (80 GB cap, 20 steps); summary in `outputs/smoke_omni/summary.txt` |

**Verified on CPU (login node):**
- `SftArguments` accept the placeholders; `eval_strategy` stays `steps`; nothing left unparsed.
- Template encode of real ST rows: **13.0 audio tokens/s** (min 12.9, max 13.2); audio and prompt tokens are masked in `labels`; the label tokens are exactly the target plus `<|im_end|>`.
- Padding-free batch of 4: `input_ids [1, 470]`, `input_features [4, 128, 717]` (padded to the longest clip in the batch, not to 30 s), `cu_seq_lens` for flash-attn; `OmniStreamingDataLoader` counted 48/48 samples.
- **Encode speed: 17 ms per sample with 1 CPU thread** (Whisper fbank dominates). With torch's default of one thread per core on the busy login node it took ~2 s per sample (thread oversubscription). DataLoader workers use 1 thread each, so training is fine; keep `OMP_NUM_THREADS` small for any main-process encoding (e.g. eval scripts).

**1-GPU smoke test, job 149688 (H200, 2026-09-28): passed.** `outputs/smoke_omni/summary.txt`:

| Run | Result |
|---|---|
| A: 40 steps, eval + checkpoint at 20/40 | loss 1.06 → 0.80; eval loss 1.342 → 1.330 (192 CoVoST2 test samples); peak 60.3 GiB; ~1.8–3.1 s/step (batch 1 × grad-accum 4, the first steps include copying and unzipping shards) |
| B: resume from A step 20 | loaded `omni_mosaic_state.json` (`sample_in_epoch` 80 = 20 steps × 4); **step 21 loss identical to A (0.5800)**; steps 22–40 within 0.02 of A (GPU non-determinism; a different batch would change the loss by far more, since per-sample losses range 0.5–1.3); saved 160 at step 40, same as A |
| C: `OMNI_GPU_MEM_GB=80` | cap 74.5 GiB (80e9 bytes, stricter than a real H100's ~79 GiB); peak 60.3 GiB, fits |

- Adapter: 388 tensors = 194 modules × 2; 4 in `audio_tower` (proj1/proj2 as intended), **0 in `audio_tower.layers`**.
- No unusable samples in any run; `epoch_size` 609,632 = the `st_v0` choose total.
- The checkpoint folder holds the usual HF files plus `omni_mosaic_state.json`.


### 10.4 Phase 7 started (2026-09-28): Megatron path, 1 GPU first

Started at the user's request, in parallel with phase 3. `train_omni.py megatron --mix <yaml> <megatron sft args>`:
- `OmniMegatronSft(MegatronSft)`: the same `_prepare_dataset` (shared `build_omni_datasets`, batch size = `--micro_batch_size`); refuses TP/PP/CP > 1 (the mosaic reader partitions by global rank; EP is fine).
- **`init_iters` fix:** Megatron's `args.init_iters()` treats `len(train_dataset)` as the *global* sample count, but mosaic's `len` is per rank. It would set `train_iters` / `eval_iters` DP× too small on multi-GPU runs, and a 1-GPU test would not show it. `run()` passes a wrapper that reports mosaic's global `epoch_size`.
- `OmniMegatronTrainer(MegatronTrainer)`: `_prepare_dataloader` → per-rank mosaic dataloaders (checks DP size == world size); `save_checkpoint` also writes `omni_mosaic_state.json`; resume (`--mcore_adapter <ckpt> --finetune false`) looks for it in the given folder, its `checkpoint-<iter>`, or a sibling `checkpoint-<iter>`. The SFT Megatron trainer doesn't read ahead (`_replace_data_iterator` is a no-op), so the dataloader's count is exact.
- Like `megatron sft`, the script sets `CUDA_DEVICE_MAX_CONNECTIONS=1`, and with `NPROC_PER_NODE` set it re-launches itself under torchrun (also used for `sft` now).
- Smoke test `slurm/smoke_megatron_1gpu.sbatch` (job 149697): MA 20 iterations (eval + resumable checkpoint at 10/20), MB resume from iteration 10, MC 80 GB cap. Attention-only LoRA (`linear_qkv linear_proj`), encoders and aligner frozen, EP=1, global batch 4.

**Megatron 1-GPU smoke, job 149697 (2026-09-28):**

| Run | Result |
|---|---|
| MA: 20 iterations, eval + checkpoint at 10/20 | **works.** loss 1.064 → (noisy, batch 4) 1.243; eval loss 0.833 → 0.821; `eval_iters` = 48 = 192 / global batch 4 (the `init_iters` fix); peak 61.5 GiB; data position 40 at iteration 10 and 80 at 20 (= 4 per iteration) |
| MB: resume from MA iteration 10 with optimizer | **failed inside Megatron-SWIFT**: `KeyError: 'optimizer'` in mcore 0.16.1 `distrib_optimizer.load_state_dict` while loading the saved optimizer state. Not our data code |
| MC: `OMNI_GPU_MEM_GB=80`, 10 iterations | works; peak 61.4 GiB (fits the 74.5 GiB cap); ~4.3 s/iteration (EP=1 on one GPU, slower than HF's ~1.8–3 s/step, as in the 8-GPU benchmark at micro-batch 1) |

- Adapter (PEFT export): 384 tensors = 48 layers × 4 attention matrices (q/k/v/o; mcore's `linear_qkv` is split back) × LoRA A/B; none in `audio_tower`.
- **Disk pitfall:** Megatron-SWIFT's default `--merge_lora true` writes a full merged model (**66 GB**) next to every checkpoint (`checkpoint-N-merged`). Always pass `--merge_lora false` for training runs; merge once at the end if needed. The smoke scripts now do; the 3 merged copies (~200 GB) were deleted.
- Follow-up job 149702 (`slurm/smoke_megatron_resume.sbatch`): MB3 resumes MA's checkpoint without the optimizer (`--no_load_optim true`) to test **our** data resume (iteration-11 loss is computed before any update, so it must equal MA's); MA2/MB2 retry the full resume with `--dist_ckpt_optim_fully_reshardable true`.

**Megatron resume follow-up, job 149702 (2026-09-28):**

| Run | Result |
|---|---|
| MB3: resume MA iteration 10, `--no_load_optim true` | **Our data resume works.** Loaded data position 40. Iteration-11 loss 1.9121 vs MA 1.9189. That batch is an outlier (other steps 0.8–1.2) and reappears at the same step; steps 12–15 each within 0.01 of MA. The small gap comes from reloading weights without the fp32 master copy; it grows to 0.14 by step 20 because Adam restarts from zero. Eval loss 0.810 at step 20 (MA 0.821) |
| MA2 → MB2: optimizer saved with `--dist_ckpt_optim_fully_reshardable true`, resumed with optimizer | Optimizer load now passes (no `KeyError`); data position 20 restored; **iteration-6 loss identical (1.1754)**. The backward pass of that step then hit `found NaN in local grad norm` and the run stopped. Forward matched exactly, so this is a Megatron-SWIFT / mcore 0.16.1 optimizer-restore bug for LoRA, not a data problem |

**Megatron optimizer resume: root cause found and fixed (jobs 149713, 149720, 2026-09-28).**
- Debug job 149713 showed:
  - plain `megatron sft` on JSONL (no `train_omni.py`) fails the same way (`KeyError: 'optimizer'`), so it is upstream;
  - `--use_distributed_optimizer false` fails too (`KeyError: 'state'`);
  - a debug hook (`OMNI_DEBUG_OPTIM_LOAD=1`) showed that the optimizer state reaching `load_state_dict` held **only** `param_state_sharding_type`, with every sharded entry missing.
- **Root cause (ms-swift 8ec0455, `swift/megatron/utils/megatron_lm_utils.py`):**
  - `load_mcore_checkpoint` builds its load template with `_generate_state_dict(args, ...)`, using the **resuming** run's args (there is a `# TODO: check no_save_optim` just above the call);
  - `_generate_state_dict` only adds the optimizer entries when `not args.no_save_optim`;
  - all our failing resumes passed `--no_save_optim true` (to save disk), so the optimizer loaded empty. That gave the two `KeyError`s, and with the fully-reshardable format a partial load and NaN gradients.
- **Fix: resume with `--no_save_optim false`** (a real run needs it anyway to resume again). Job 149720:
  - default format: iterations 11–20 within **0.009** of the uninterrupted run (vs 0.136 without the optimizer); eval loss 0.8219 vs 0.8211;
  - fully-reshardable format: no NaN, iteration 6 loss identical (1.1754), max difference 0.015.
  - This matches the HF resume.
- `train_omni.py` now refuses `--finetune false` + optimizer loading + `--no_save_optim true` with an explanation (`check_megatron_resume_args`).
- An upstream issue draft is in `docs/upstream_issue_megatron_resume.md` (not posted).
- `--no_load_optim true` remains an option when the optimizer state is not wanted.

### 10.5 Checkpoint / disk policy (audit 2026-09-28)

Audit of every training and test script (`slurm/*.sbatch`, `train_lora_asr_smoke.sh`, `container/smoke_test.sh`, `container/bench_8gpu/bench_8gpu.sbatch`) and of everything written to `/scratch/prj0000000234/sailorhb`:
- The only large dumps were Megatron-SWIFT's `checkpoint-N-merged` folders (66 GB each, from its default `--merge_lora true`) written by the first Megatron smoke test: 3 copies, ~200 GB, all deleted. No files over 5 GB were written anywhere else (apart from the model weights).
- Every Megatron command in the scripts now passes `--merge_lora false`. The older container smoke test and the 8-GPU benchmark already did.
- HF `swift sft` with LoRA saves adapters only (merging happens only in `swift export --merge_lora`). Checkpoint sizes: HF LoRA ~150 MB with optimizer; Megatron LoRA ~165 MB (PEFT adapter + ~140 MB mcore `iter_*` with optimizer).
- Current outputs total ~3 GB (`outputs/`, `container/bench_8gpu/out`, `container/smoke_outputs`).

**Guards in `train_omni.py`** (`safe_save_defaults`, logged, and an explicit flag always wins):
- megatron: `--merge_lora false`;
- both modes: `--save_total_limit 3` (Megatron-SWIFT and HF otherwise keep every checkpoint).

**Rules for real runs:**
- Never merge LoRA during training. Merge at the end (`swift export --merge_lora`) for deployment. **Correction 2026-09-29:** vLLM 0.17.1 can *not* serve LoRA adapters for Qwen3-Omni (`DEPLOYMENT.md` §1), so the shipped model is always a merged copy.
- Megatron: `--save_safetensors true` gives a PEFT adapter / HF weights; add `--no_save_optim false` only when the run must be resumable, and keep `--save_total_limit` small.
- Full fine-tuning checkpoints are large (HF-format weights ~61 GB; Megatron with optimizer state ~490 GB for 30.5 B params). Use sparse `--save_steps`, a small `--save_total_limit`, and check free space on `/scratch` first (86% used on 2026-09-28).
- The eval script (phase 5) uses transformers + PEFT for in-training checks. For final numbers it may merge a checkpoint for vLLM, but must delete the merged copy (~66 GB) after scoring and keep at most one at a time.

### 10.6 Multi-GPU check on 4 GPUs (2026-09-29): both paths pass

The 8-GPU job 149729 waited in the queue (no node had 8 free GPUs), so the same test ran on 4 GPUs first: `slurm/train_4gpu_both.sbatch`, **job 150431**, node 5FS9B74, 24 min.
- It is a copy of `train_8gpu_both.sbatch` with 4 GPUs, 32 CPUs and 500 GB, and output in `outputs/train_4gpu/`.
- **Global batch kept at 32:** HF is 1 × 4 GPUs × grad-accum 8 (was 4); Megatron uses EP=4 (was 8) with global batch 32.
- Runs are named H4A/H4B/M4A/M4B and use ports 2958x.

| | HF DDP (`sft`) | Megatron EP=4 |
|---|---|---|
| Train loss, step 1 → 60 | 1.175 → 0.831 | 1.172 → 0.837 |
| Eval loss, step 30 / 60 (192 CoVoST2 test samples) | 1.314 / 1.312 | 0.804 / 0.801 |
| Peak memory per GPU | 61.8 GiB | 23.5 GiB |
| s/step (last) | 4.8 | 5.6 |
| Resume from step 30: max \|loss diff\| over steps 31–60 | 0.004 (step 31 identical) | 0.003 (with optimizer) |
| Data position saved at step 30 / 60 | 960 / 1920 | 960 / 1920 |
| Rank partitions (`check_partitions.py`) | 4 × 488 ids, disjoint, no repeats | same |
| Adapter | 388 tensors, 0 in `audio_tower.layers` | 384 tensors, 0 in `audio_tower` |

Notes:
- **Megatron `eval_iters` = 6 = 192 / 32** on DP=4. This confirms the `_GlobalLen` fix (D16) on more than one GPU.
- **The eval-loss gap between trainers (1.31 vs. 0.80) is not new.** The 1-GPU smoke tests showed the same gap (1.33 vs. 0.82). It comes from how each trainer averages eval loss; train losses agree.
- Each rank read 488 samples (1952 in total) for 1920 trained. Both trainers show the same count, so it is most likely dataloader prefetch. Not investigated.
- **H100:** Megatron at 23.5 GiB per GPU (EP=4) fits an 80 GB H100 easily.

### 10.7 LoRA inside the audio encoder (2026-09-29)

**New options in `train_omni.py`** (both `sft` and `megatron`):

| Option | Values | Default |
|---|---|---|
| `--audio_lora_layers` | `none` \| `all` \| `top<K>` (last K layers) \| `bottom<K>` \| `<a>-<b>` \| `i,j,k` (0-based, ranges inclusive; 32 layers, count read from the model config) | `none` |
| `--audio_lora_modules` | `attn` (`self_attn.q/k/v/out_proj`) \| `mlp` (`fc1`, `fc2`) \| `attn_mlp` | `attn` |

```bash
python train_omni.py sft      ... --audio_lora_layers top8 --audio_lora_modules attn_mlp \
    --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) ...
python train_omni.py megatron ... --audio_lora_layers all --target_modules linear_qkv linear_proj ...
```

**How it works** (`add_audio_lora_targets`, `fix_peft_target_modules`):
- **Target names.** The script adds names such as `audio_tower.layers.31.self_attn.q_proj` to the existing `--target_modules` list. PEFT matches a target when the module name *ends with* it. HF names the module `thinker.audio_tower...` and Megatron-SWIFT names it `visual.audio_tower...` (mcore_bridge `Qwen3Omni_Vit`), so one list works for both trainers.
- **Why the names are inserted, not added as a flag.** ms-swift parses `--target_modules` as one list, and a second `--target_modules` flag *replaces* the first.
- **Refused combinations** (the run exits with an error):
  - no explicit `--target_modules`;
  - `all-linear` in the list: ms-swift then turns the list into a regex and silently drops the extra names;
  - `--target_regex`.
- **Gradients reach the encoder with no patch.**
  - ms-swift turns on encoder gradient checkpointing with `enable_input_require_grads`.
  - Under DDP it uses `use_reentrant=False`.
  - `patch_frozen_module` keeps autograd when any encoder parameter is trainable.
- **Exact adapter configs.** With encoder LoRA on, every save rewrites `adapter_config.json` → `target_modules` to the exact modules that have weights in `adapter_model.safetensors`; the original is kept as `adapter_config.orig.json`. This is needed because both trainers write names that match more modules than were trained:
  - Megatron writes bare names (`q_proj`, `fc1`, ...), which match all 32 encoder layers and the LLM;
  - swift sft shortens the list (e.g. `28.self_attn.k_proj`, which also matches LLM layer 28).

  PEFT would wrap those extra modules in zero LoRA at load time. That is harmless, but it costs memory and makes the adapter misleading. Megatron resume doesn't read `target_modules` from this file (only `r`, `lora_alpha`, `lora_dropout`, `use_rslora`, `bias`), so resume is unaffected.
- **Without the flags,** argv and saves are unchanged.

**Test:** `slurm/smoke_audio_lora_2gpu.sbatch`, **job 150442** (2 GPUs; top4 × attn_mlp; 20 steps each). `tests/summarize_smoke.py` now lists the encoder layers and modules that have LoRA and how many `lora_B` are non-zero.

| | HF DDP | Megatron EP=2 |
|---|---|---|
| Encoder layers with LoRA | 28–31 only | 28–31 only |
| Modules per layer | q/k/v/out_proj, fc1, fc2 | same |
| Encoder `lora_B` non-zero after training | **24/24** | **24/24** |
| Loss, step 1 → 20 | 1.030 → 0.726 | 1.033 → 0.722 |
| Peak memory per GPU | 61.8 GiB (unchanged) | 34.4 GiB |
| `adapter_config.json` target_modules | 218 exact names | 216 exact names |

- The first attempt (job 150441) failed only on the Megatron side, because the test passed `--save_total_limit 1`. Megatron-SWIFT requires ≥ 2.
- **Not yet tested:** loading these adapters with PEFT for inference, and merging them (`swift export --merge_lora`). vLLM 0.17.1 can't serve *any* LoRA for Qwen3-Omni (`DEPLOYMENT.md` §1), so serving goes through a merged model.
- Suggested settings from `FINETUNING_GUIDE.md` §5.1: `top8`, r=16 for AEC/SER; `top16` or more, r=32 for new languages.

### 10.8 Moving to the H100 cluster (2026-09-29)

`transfer/` holds everything needed to copy the toolkit to another cluster (ASPIRE2A+). See `transfer/TRANSFER_TO_H100.md`:
- `copy_to_h100.sh`: rsync, dry run by default. Parts are code, container, model and data (~275 GB).
- `relocate_paths.sh`: rewrites the hard-coded `/scratch/...` paths and the container bind mounts once on the new cluster. Tested on a local copy: no `/scratch/` references were left.
- `requirements.txt`, `requirements_pydeps.txt`, `requirements_container_freeze.txt`: pinned packages. They are only needed if the `.sif` can't be used.
- **Not copied:** `toolkits/venvs/swift_omni` (a Python 3.10 venv on `nemo_env` in `$HOME`, not portable) and `outputs/`.

### 10.9 Deployment check (2026-09-29): vLLM can't serve our adapters

- **`supports_lora(Qwen3OmniMoeThinkerForConditionalGeneration)` is False in vLLM 0.17.1** (True for Qwen2.5-Omni). This affects every LoRA we train, not only the audio-encoder LoRA.
- **Consequences** (details in the new **`DEPLOYMENT.md`**):
  - deployment ships a *merged* model that stock vLLM serves;
  - eval uses transformers + PEFT, or merges temporarily for vLLM;
  - multi-LoRA serving for the two-track plan is not available.
- **Corrected in this file:** §2 diagram, §3.3, §4, §5, §10.5, §10.7.
- **Rule for training:** changes must stay at the weight level (LoRA → merge, or full fine-tuning), so the architecture matches the public model (`DEPLOYMENT.md` §5).

### 10.10 Version control (2026-09-29)

- **Repository:** `git@github.com:HardikSailor/qwen3_omni_speechllm.git`.
  - `.gitignore` excludes `outputs/`, logs, checkpoints and weights, generated `data/`, audio, and caches.
  - The first commit is about 100 files, ~830 KB.
- **ms-swift:** git submodule `third_party/ms-swift`, pinned at `8ec045582` = the commit in the container (D23).
  - Clone with `--recursive`.
  - Upgrades go through a branch, a container rebuild and the smoke tests.

### 10.11 8-GPU check (2026-09-29): both paths pass, throughput measured

Job **149729** (`slurm/train_8gpu_both.sbatch`, node 374Y574, 8×H200, 20 min) finally ran. Settings as in §10.6 but on 8 GPUs: HF DDP (micro-batch 1 × 8 GPUs × grad-accum 4) and Megatron **EP=8** (micro-batch 1, global batch 32); `mixes/st_v0.yaml`; 60 steps; eval + checkpoint at 30/60. Summary: `outputs/train_8gpu/summary.txt`.

| | HF DDP (`sft`) | Megatron EP=8 |
|---|---|---|
| Train loss, step 1 → 60 | 1.175 → 0.835 | 1.172 → 0.836 |
| Eval loss, step 30 / 60 | 1.310 / 1.305 | 0.805 / 0.803 |
| Peak memory per GPU | 61.8 GiB | **16.7 GiB** |
| **Steady throughput** (steps 32–59, no eval/save in the window) | **2.19 s/step = 14.6 samples/s** | **2.30 s/step = 13.9 samples/s** |
| Resume from step 30: max \|loss diff\| over steps 31–60 | 0.004 (step 31 identical) | 0.004 (with optimizer, `--no_save_optim false`) |
| Data position saved at step 30 / 60 | 960 / 1920 (= 30 × 32 / 60 × 32) | 960 / 1920 |
| Rank partitions (`check_partitions.py`) | 8 × 248 ids, disjoint, no repeats | same |
| Adapter | 388 tensors, 0 in `audio_tower.layers` | 384 tensors, 0 in `audio_tower` |

Notes:
- **Megatron `eval_iters` = 6 = 192 / 32 on DP=8**: the global-length fix (D16) holds at 8 GPUs.
- **Scaling 4 → 8 GPUs:** HF 7.6 → 14.6 samples/s (1.92×); Megatron EP 6.9 → 13.9 (2.0×). Near-linear, so the mosaic data path keeps up at this scale.
- **Compared with the LibriSpeech JSONL benchmark (job 147727: 17.0 vs. 13.5 samples/s):** on the real ST mix HF is a little slower (longer clips: People's Speech ~15 s, GigaSpeech up to 10 s, vs. LibriSpeech) and the HF–Megatron gap shrank from 26% to 5%. Megatron uses a quarter of the memory, so bigger batches or packing (still to test) could put it ahead.
- **How the numbers were computed:** `train_speed(s/it)` in the logs is a running average that includes start-up, eval and saving (HF: 7.1 at step 6, 2.8 at step 60). The steady figure is the `elapsed_time` difference between steps 32 and 59 divided by 27.
- Each rank read 248 ids for 240 trained samples, the same small prefetch overshoot seen on 4 GPUs.
- Outputs: 964 MB (`--merge_lora false`, `--save_total_limit 2`).


### 10.11 enroot support (2026-09-29)

The new server has enroot, not Apptainer:
- `container/sif_to_enroot.sh` converts the image unchanged (→ `.sqsh`, 12.6 GB).
- `run_container.sh` falls back to `run_container_enroot.sh`, so job scripts are unchanged.
- **Job 150477:** the audio-LoRA smoke test passed on both trainers under enroot, with the same losses and memory as under Apptainer. Details: D24, `container/README.md`.

### 10.12 Multi-node on NSCC and W&B logging (2026-10-01)

Reserved queue `R212478` (project `13003558_R4`, 12 DGX H100 nodes), the only GPU queue from now on (D26).
`pbs/train_16gpu_2node.pbs` and `pbs/train_32gpu_4node.pbs`: NCCL check, then both trainers on the ST mix, global batch 32,
60 steps with eval and checkpoints at 30 / 60, resume 30 -> 60, partition check.

| | NCCL busbw | HF DDP s/step | Megatron EP=8 s/step | Peak GiB HF / Megatron | Resume diff HF / Megatron |
|---|---|---|---|---|---|
| 2 nodes, 16 GPUs (job 215070) | 416 GB/s | 1.45 | 2.2 | 61.7 / 14.1 | 0.004 / 0.003 |
| 4 nodes, 32 GPUs (job 215088) | 308 GB/s | 0.91 | 1.64 | 61.7 / 14.0 | 0.004 / 0.005 |

- Partitions disjoint on 16 and 32 ranks; the data position (960 samples at step 30) restored on resume.
- Eval loss on the fixed validation set matches across 2 and 4 nodes (HF 0.82-0.83, Megatron 0.80).
- Fixed on the way: `pbsdsh` re-parsed the phase command (B23); the reservation refuses `place=scatter:excl`.
- W&B (D27): `--wandb true` / `qsub -v WANDB=1`; all runs of these jobs are in `i2r-llm/meralion_v4`.

