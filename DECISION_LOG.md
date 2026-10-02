# Qwen3-Omni fine-tuning: how we got here (decision log)

> **Cluster note (2026-09-30):** this document was written on the previous cluster (Orion: Slurm, H200, Apptainer). The project now runs on NSCC (PBS, H100, enroot). Map the old paths with the *Paths* section of `README.md`; the container is described in `container/README.md`. Job numbers (e.g. 150431) are Orion Slurm jobs.

Written 2026-09-28. It covers the work from 2026-09-24 to 2026-09-28 and records **what** we decided, **why**, and **what evidence** we had at the time. Detailed designs live in the documents listed in §1. This file links them together and does not repeat them.

Conventions:
- **D-numbers** are decisions (§3).
- **B-numbers** are problems found and fixed (§4).
- Slurm job IDs are given so logs can be found again.
- "User decision" marks choices the user made. Everything else was a recommendation that the user accepted, or a technical fix.

---

## 1. Document map

| Document | What it holds |
|---|---|
| `FINETUNING_GUIDE.md` | The research plan: SEA/SG multi-task data, stages (Stage 0 language adaptation → joint → speech / acoustic-event tracks), architecture ideas, RL, evaluation. §12: training stack (HF vs. Megatron) |
| `TRAINING_AND_LOCAL_VOCAB_EXPERIMENTS.md` | Grounded training and local-vocabulary (contextual biasing) experiments |
| `MDS_DATA_PIPELINE.md` | How our MDS data is fed to ms-swift. §0 old vs. new; §0.5 streaming vs. map-style (and the decision); §1 what the data is; §2–§8 the detailed design |
| `PIPELINE_PLAN.md` | The pipeline we are building: where every piece of code comes from (§3), layout (§4), build phases (§6), open decisions (§7), parallelism (§8), H100 portability (§9), progress and results (§10) |
| `WORKLOG.md` | Day-by-day log |
| **`DECISION_LOG.md`** (this file) | Why things are the way they are |
| `omni_mds/VENDORED.md` | Which files were copied from the old trainer, with md5s and the one local patch |
| `docs/upstream_issue_megatron_resume.md` | Draft bug report for ms-swift (not posted) |
| `DEPLOYMENT.md` | Serving and partner on-prem hand-off: vLLM can't serve Qwen3-Omni LoRA, merged-model deployment, partner requirements, sizing, package, training rules, licences (2026-09-29) |
| `transfer/TRANSFER_TO_H100.md` | What to copy to the H100 cluster, the copy and path-rewrite scripts, requirement files (2026-09-29) |
| `/scratch/prj0000000234/sailorhb/container/README.md` | The training container |
| Shared doc "Megatron-SWIFT vs swift sft for Qwen3-Omni" | https://claude.ai/code/artifact/c06ff669-814a-4abf-8feb-de36b7f4ea8d (2026-09-24) |

---

## 2. Timeline

**2026-09-24 (earlier sessions)**
1. Downloaded Qwen3-Omni-30B-A3B-Instruct. Tested two conda envs and picked `nemo_env` (D1).
2. Created a LoRA fine-tuning venv (`venvs/swift_omni`, ms-swift over nemo_env). A LibriSpeech smoke test exposed the PEFT regex leak (B1) and the target-style lesson (D3).
3. Wrote `FINETUNING_GUIDE.md` for the SEA/SG data scope.
4. Assessed Megatron. Raw Megatron-LM cannot load the model; Megatron-SWIFT can (D4).
5. Built and smoke-tested the training container `swift_megatron_cu128.sif`.
6. Submitted the 8-GPU trainer benchmark, job 147727.
7. Studied the old `multimodal_trainer` data code and the MDS shards, and wrote `MDS_DATA_PIPELINE.md`, first recommending a map-style reader.

**2026-09-28 (this session)**
1. Read the benchmark result: swift DDP 17.0 samples/s vs. Megatron EP=8 13.5 samples/s (D5).
2. Asked: can the old `multimodal_trainer` be used? Answer: keep its data, text and eval parts; drop its trainer and model code (D6).
3. Compared the old and new dataset handling (`MDS_DATA_PIPELINE.md` §0).
4. Questions on unzip time, Hugging Face copies of the datasets, and where the data lives (D7). Submitted the unzip benchmark (job 149550).
5. The user proposed "old streaming data + ms-swift training". Showed that streaming doesn't avoid unzipping, and that it should be a plug-in, not a new training loop (D8). Wrote `MDS_DATA_PIPELINE.md` §0.5.
6. Wrote `PIPELINE_PLAN.md`, then added parallelism (D9) and H100 support. The user confirmed ASPIRE2A+ (D10).
7. **Phase 1:** ported the old per-sample text logic for all tasks (the user asked for all, D11) and made the ST-only pilot mix (user decision, D13). The port matches the old code exactly.
8. User decisions: ASR normalisation off (D12); reader = mosaic streaming (D8).
9. **Phase 2:** mosaic reader; 8/8 tests.
10. **Phase 3:** `train_omni.py sft` plus the 1-GPU smoke test (job 149688) passed (D15).
11. **Phase 7 (Megatron), started at the user's request:** `train_omni.py megatron`; `init_iters` fix (D16); 1-GPU smoke test (job 149697).
12. Found the Megatron `merge_lora` disk issue. At the user's request, audited all scripts and added a disk policy (D18).
13. Megatron optimizer resume: the user asked to fix it first. Root cause found in ms-swift and verified fixed (D17; jobs 149702, 149713, 149720).
14. Unzip benchmark 149550: about 9 min per node for the full mix.
15. Submitted the 8-GPU job for both paths (job 149729).

**2026-09-29**
1. Job 149729 was still pending (no node with 8 free GPUs). At the user's request, ran the same test on 4 GPUs (`slurm/train_4gpu_both.sbatch`, job 150431). Both paths passed (D19).
2. At the user's request, prepared the move to the H100 server: requirement files, a copy script and a path-rewrite script in `transfer/` (D21).
3. At the user's request, added LoRA inside the audio encoder: all layers, top-K or any layer set (D20). Job 150441 failed on a test-script flag (B19); job 150442 passed for both trainers.
4. The user asked what a partner company would need to run the model on-prem. Checking vLLM showed it can't serve LoRA for Qwen3-Omni (B20). Wrote `DEPLOYMENT.md` and corrected the docs that assumed it could (D22).
5. Put the project under git (`README.md`, `.gitignore`; first push by the user). Added ms-swift as a submodule pinned at `8ec0455` (D23).
6. The user said the new server has enroot, not Apptainer. Converted the image to enroot and added a launcher that is picked automatically; tested end to end (D24).

---

## 3. Decisions

### D1. Python environment for Qwen3-Omni: `nemo_env` (2026-09-24, user decision)
- **Options:** `nemo_env` (torch 2.6+cu124, flash-attn 2.7.4, transformers 5.8.dev) or `py2hf` (transformers 4.57, SDPA only).
- **Evidence:** all tests passed on an H200 in both. `nemo_env` loads in 20 s vs. 57 s and has flash-attention; about 66 GiB in bf16.
- **Consequences:**
  - `nemo_env` was made self-contained (`PYTHONNOUSERSITE=1`, packages moved out of `~/.local`).
  - **Never install CUDA-13 nvidia wheels into it:** they overwrite torch's cuDNN and break audio (`CUDNN_STATUS_NOT_INITIALIZED`).
  - Package list: `env_backup/nemo_env_final_freeze.txt`.

### D2. LoRA tooling: ms-swift, in a venv on top of nemo_env (2026-09-24)
- **Why a venv:** ms-swift needs `datasets<4.8.5`, but nemo_env has 5.0. Cloning the conda env stalled on `/home` I/O for over 20 min.
- **Setup:** `venvs/swift_omni` uses `--system-site-packages` with constraints (`env_backup/swift_constraints.txt`), so pip cannot replace torch or transformers.
- Superseded for multi-GPU work by the container (D4). The venv is kept for 1-GPU debugging.

### D3. Train on natural, cased, punctuated targets (2026-09-24, evidence-based)
- **Evidence:** 60 LoRA steps on raw LibriSpeech targets (upper case, no punctuation) made the model copy that style, and dev-other WER got worse (2.98% → 3.97%).
- **Consequence:** the model learns target *style*, not just content. This led to D12 (ASR normalisation off) and to the still-open decision on the target convention (speaker tags, hashtags).

### D4. Scale-out stack: Megatron-SWIFT in a container, not raw Megatron-LM (2026-09-24)
- **Evidence:**
  - the local Megatron-LM (core 0.20) has no Qwen3-Omni model, no audio encoder and no converter;
  - Megatron-SWIFT (ms-swift + mcore-bridge) supports Qwen3-Omni;
  - Megatron-SWIFT needs megatron-core ≥0.16,<0.20, Transformer Engine and apex, and nemo_env must not get more CUDA wheels (D1).
- **Decision:** the container `/scratch/prj0000000234/sailorhb/container/swift_megatron_cu128.sif`:
  - ms-swift 4.6.0.dev0 @ 8ec0455, mcore 0.16.1 (0.17+ needs Python 3.12), TE 2.13, torch 2.10+cu128, vLLM 0.17.1;
  - launched with `run_container.sh`;
  - smoke tests pass for inference, vLLM, `swift sft` and `megatron sft`.

### D5. Which trainer for LoRA (job 147727, read 2026-09-28)
- **Benchmark:** 8×H200, LoRA r16 attention-only, 32 samples/step, micro-batch 1, LibriSpeech JSONL:

  | Trainer | Samples/s | Memory per GPU |
  |---|---|---|
  | swift sft DDP (grouped_mm and default experts: same) | **17.0** | 61.8 GiB |
  | megatron sft EP=8 | 13.5 | **14.2 GiB** |

- **Decision:**
  - use HF `swift sft` for LoRA prototyping;
  - use Megatron for expert LoRA, full fine-tuning and multi-node;
  - Megatron needs a fair re-run with bigger batches or packing: it had a quarter of the memory in use, so this benchmark does not show its best case.

### D6. The old `multimodal_trainer`: reuse data knowledge, not the trainer (2026-09-28)
- **Question:** can the past member's toolkit train Qwen3-Omni?
- **Finding:**
  - it is built around the MERaLiON encoder + adapter + LLM design;
  - it is pinned to transformers 4.44 and PEFT 0.11 (Qwen3-Omni needs ≥4.57);
  - its FSDP only shards dense models and has no expert parallelism;
  - a Phi-4 wrapper shows a single-model integration is possible, but we would re-implement what ms-swift already does and has tested.
- **Decision:**
  - **keep:** MDS data, mix YAMLs, text normalisers, the multilingual instruction library, prompt and answer clean-up, augmentation, metrics;
  - **drop:** collators, trainer, FSDP, model code;
  - **never write our own training loop** (principle, `PIPELINE_PLAN.md` §1).

### D7. Data source: the existing MDS shards, as they are (2026-09-28)
- **Location:**
  - `/scratch/prj0000000234/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal/` (owned by `zoux`; read only for us);
  - 117 of the 195 training paths in the v4.3 mix are readable: 81.9 M samples, 1.81 TB zstd, 2.01 TB unzipped;
  - 78 are missing (71 wonghmj SQA and 7 wonghmj AC).
- **Rejected:**
  - *Export to JSONL/parquet*: ~82 M files or +2 TB on a busy `/scratch`.
  - *Hugging Face copies of the datasets*: about half the mix by size (IMDA NSC, vendor sets) is not on HF. The public sets there are not the same data (ours are already cut to ≤30 s, normalised and turned into instruction/ST/SQA/SDS pairs). `load_dataset` also writes a full Arrow copy. HF stays useful for *adding* public sets.
- **Measured (job 149550):** zstd saves only 2% (Opus) to 22% (WAV). Unzipping the full mix takes ~9 min per node (see D8).

### D8. Reader: mosaic streaming plugged into ms-swift (2026-09-28, user decision)
- **Options:**
  - D, map-style: unzip once, build an index, random access;
  - C, the old trainer's mosaic `StreamingDataset` as a plug-in.
- **Key facts:**
  - **Streaming does not avoid unzipping.** Mosaic unzips each shard into its local cache as well, just lazily and again each job.
  - ms-swift's built-in `--streaming` makes rank 0 read and scatter every batch, which does not scale.
  - Both trainers have one clean dataloader method to override.
- **User's proposal:** "old streaming data + ms-swift training". Agreed as a plug-in, not a new training loop.
- **Decision rule offered:** D if a permanent unzipped copy is acceptable, else C. The user chose **C**.
- **Later evidence (job 149550):**
  - reading from `/scratch` at 4.45 GB/s and unzipping to local NVMe at 3.3 GB/s (32 in parallel), i.e. ~9 min per node for the full mix;
  - training reads only a few MB/s, so mosaic's lazy unzip is never a bottleneck.

### D9. Parallelism (2026-09-28)
- **Model size (from the safetensors headers):**
  - thinker 31.72 B parameters = 59.1 GiB in bf16;
  - 91% of it is experts (28.99 B);
  - the talker and code2wav are not loaded for training.
- **Decision:**
  - DDP for LoRA (the frozen weights fit on one GPU);
  - Megatron EP for expert LoRA and full fine-tuning (EP=8 within a node, data parallel across nodes, distributed optimizer);
  - FSDP1 is out (no ms-swift preset, legacy);
  - FSDP2 / ZeRO-3 only as a fallback (they don't split experts; ms-swift's own benchmark: ZeRO-3 is ~10× slower than Megatron for Qwen3-30B-A3B full fine-tuning).

### D10. Must also run on ASPIRE2A+ H100 80 GB (2026-09-28, user requirement; cluster confirmed by the user)
- **Same Hopper architecture, so only memory differs:**
  - DDP LoRA fits but is tight (61.8 of ~79 GiB);
  - EP=8 is the default on H100;
  - full fine-tuning needs ≥2 nodes (estimate);
  - vLLM needs TP=2.
- **Rules:**
  - no hardware constants in code;
  - per-cluster profiles;
  - the same global batch on both clusters;
  - drop samples that are too long before encoding;
  - emulate the H100 memory with `OMNI_GPU_MEM_GB=80` (used in every smoke test).
- **Open:** Apptainer vs. enroot on ASPIRE2A+, whether the data is reachable from there, local disk size (`PIPELINE_PLAN.md` §9.4).

### D11. Port the old text logic for all tasks, with exact parity (2026-09-28; "all tasks" was the user's request)
- `omni_mds/sea_text.py` `build_row` turns one MDS sample into an ms-swift row. The old modules are copied unchanged (`omni_mds/VENDORED.md`).
- **Evidence:** prompts and targets are identical to the old `InstructDataCollator` on real rows with the same seed:
  - **8,392/8,392** in total;
  - ST 2,792, ASR 2,000 (10 datasets covering every language branch), PQA/SDS/CPQA/AQA/AC 1,600;
  - each under 4 configurations.
- **Deliberate differences, each tested:**
  - a per-sample RNG;
  - AC prompts use `ac_prompt_type` (the old code used the ASR setting);
  - no MERaLiON system prompt and no `<SpeechHere>` template;
  - `context_text` is never put in the prompt (it would leak the answer for ST/ER).

### D12. ASR normalisation off by default, kept as an option (2026-09-28, user decision)
- The old default lower-cased ASR targets and stripped punctuation. It is now off because of D3.
- `RowConfig(asr_normalize_text=True)` / `--asr_normalize_text true` restores the old behaviour.

### D13. Start with an ST-only pilot mix (2026-09-28, user decision)
- `mixes/st_v0.yaml`: all 8 ST training sets (~610 k samples/epoch via `choose`) and 6 CoVoST2 validation sets.
- ST is the simplest task for parity: the old code passed ST targets through unchanged and used the dataset's own instruction.
- **Notes:**
  - the ST "validation" sets are the CoVoST2 **test** split;
  - the `language` column is empty for ST, so languages are set in the YAML.

### D14. Instruction dropout keeps the audio (2026-09-28, parity)
- The first port dropped the audio for dropout rows. Re-reading the old code showed it kept the (unrelated) audio.
- The default is now the old behaviour; `dropout_keep_audio=False` gives text-only rows. Covered by a parity test.

### D15. Entry-script design: subclass ms-swift, don't patch it (2026-09-28)
- `train_omni.py`:
  - subclasses `SwiftSft` / `MegatronSft` (`_prepare_dataset`);
  - wraps the trainer class for per-rank mosaic dataloaders;
  - saves `omni_mosaic_state.json` with every checkpoint;
  - on resume, restores the data position and sets `ignore_data_skip`: HF's own skipping would re-read and re-encode every skipped sample.
- ms-swift requires `--dataset` and needs `--val_dataset` to keep evaluation on, so the script passes placeholders that are never loaded.
- With `NPROC_PER_NODE` set it re-launches itself under torchrun, like the swift/megatron CLIs.
- **Evidence (HF, job 149688):**
  - training and eval work;
  - resume: step-21 loss identical, and the saved data position is exact;
  - peak 60.3 GiB, also under the 80 GB cap;
  - no LoRA in `audio_tower.layers`.

### D16. Megatron needs the *global* dataset length (2026-09-28)
- Megatron's `args.init_iters()` treats `len(train_dataset)` as the global sample count, but mosaic's `len` is per rank.
- `train_iters` and `eval_iters` would have been DP× too small on multi-GPU runs, and a 1-GPU test cannot show this. It was found by reading the code, before any multi-GPU run.
- **Fix:** a `_GlobalLen` wrapper reporting mosaic's `epoch_size`.
- **Evidence:** job 149697 set `eval_iters` = 48 = 192 / 4 on 1 GPU; jobs 150431 (DP=4) and 149729 (DP=8) set 6 = 192 / 32.

### D17. Megatron optimizer resume: root cause in ms-swift; fix = `--no_save_optim false` when resuming (2026-09-28; "fix it first" was the user's call)
- **Symptoms:**
  - default format: `KeyError: 'optimizer'`;
  - `--use_distributed_optimizer false`: `KeyError: 'state'`;
  - `--dist_ckpt_optim_fully_reshardable true`: NaN gradients after loading.
- **Evidence it is upstream:** plain `megatron sft` on JSONL fails the same way (job 149713).
- **What the debug hook showed:** the optimizer state reaching `load_state_dict` held only `param_state_sharding_type`.
- **Root cause:**
  - `load_mcore_checkpoint` builds its load template from the **resuming** run's args;
  - `_generate_state_dict` adds optimizer entries only when `not args.no_save_optim`;
  - our resumes passed `--no_save_optim true` to save disk.
- **Evidence the fix works (job 149720):**
  - default format: iterations 11–20 within 0.009 of the uninterrupted run (vs. 0.136 without the optimizer);
  - fully-reshardable format: no NaN, identical first loss.
- **Consequences:**
  - `train_omni.py` refuses the bad combination;
  - upstream issue draft in `docs/upstream_issue_megatron_resume.md`;
  - `--no_load_optim true` remains an option for resuming without the optimizer.

### D18. Disk and checkpoint policy (2026-09-28, user request)
- **Trigger:** Megatron-SWIFT's default `--merge_lora true` wrote a **66 GB** merged model next to every checkpoint. Three copies (~200 GB) were deleted.
- **Audit:** no other script dumps large files. All outputs total about 3 GB.
- **Guards:** `train_omni.py` adds `--merge_lora false` (Megatron) and `--save_total_limit 3` unless they are given explicitly.
- **Rules (`PIPELINE_PLAN.md` §10.5):**
  - never merge LoRA per checkpoint during training. *(The original reason, "vLLM loads adapters", turned out to be wrong for Qwen3-Omni; see D22. Eval merges are temporary and deleted after scoring.)*
  - save optimizer state only when a run must be resumable;
  - full fine-tuning checkpoints are huge (~490 GB with optimizer), so save rarely and check free space.

### D19. 4-GPU runs keep the 8-GPU global batch (2026-09-29, user request)
- **Trigger:** the 8-GPU job 149729 was pending (priority and resources), while three nodes each had 5 free GPUs.
- **Decision:** `slurm/train_4gpu_both.sbatch` is the same test on 4 GPUs.
  - Global batch stays 32: HF grad-accum 8 instead of 4; Megatron EP=4 with global batch 32.
  - Losses are therefore comparable with the 8-GPU run.
- **Result (job 150431):** everything passes (`PIPELINE_PLAN.md` §10.6):
  - resume within 0.004 (HF) and 0.003 (Megatron);
  - disjoint partitions;
  - `eval_iters` = 6 = 192/32, confirming D16 on DP > 1;
  - Megatron EP=4 at 23.5 GiB per GPU.
- The 8-GPU job was left queued for EP=8 throughput.

### D20. LoRA inside the audio encoder, selectable per layer (2026-09-29, user request)
- **Options:** `--audio_lora_layers none|all|top<K>|bottom<K>|a-b|i,j,k` and `--audio_lora_modules attn|mlp|attn_mlp`. They work in both trainers and are off by default, so existing runs are unchanged.
- **Design:**
  - Suffix names (`audio_tower.layers.N.<linear>`) are inserted into the existing `--target_modules` list, so the same list works for HF and Megatron (B17).
  - Combinations where ms-swift would silently drop the names are refused (B18).
  - Configs are exact: after each save, `adapter_config.json` lists only the modules that have weights (B17).
- **Why not `--freeze_vit false` + `all-linear`:** that puts LoRA on every linear of both the audio *and vision* encoders, with no per-layer choice.
- **Evidence (job 150442):** on both trainers, LoRA landed only in layers 28–31 and all 24 encoder `lora_B` weights trained. `PIPELINE_PLAN.md` §10.7.
- **Open:** loading these adapters with PEFT and merging them are not yet tested. vLLM can't serve them unmerged (D22).

### D21. Moving to the H100 cluster: copy the image, not a Python env (2026-09-29, user request)
- The environment is the Apptainer image `swift_megatron_cu128.sif` plus `toolkits/pydeps`.
- `toolkits/venvs/swift_omni` can't be moved: it is a Python 3.10 venv on the `nemo_env` conda env in `$HOME`.
- **Files in `transfer/`:**
  - `copy_to_h100.sh`: code, container, model and data, ~275 GB; rsync, dry run by default;
  - `relocate_paths.sh`: rewrites the `/scratch/...` paths and bind mounts; tested on a local copy;
  - `requirements*.txt`: pinned top-level packages, the pydeps extras, and the image's full `pip freeze`, for a rebuild without Apptainer;
  - `TRANSFER_TO_H100.md`: the guide.
- **Still to confirm on ASPIRE2A+:** Apptainer/Singularity availability, PBS vs. Slurm headers, and whether the MDS data is already there.

### D22. Deployment = merged model on stock vLLM; keep the architecture stock (2026-09-29, from the user's on-prem question)
- **Finding (B20):** vLLM 0.17.1 `supports_lora(Qwen3OmniMoeThinkerForConditionalGeneration)` = False.
- **Decisions:**
  - Ship a merged Hugging Face model (`swift export --merge_lora`). The partner needs only a GPU host with an NVIDIA driver, Docker and the pinned official vLLM image; none of our code.
  - Eval: transformers + PEFT during training; merge → vLLM → delete for final numbers.
  - Training changes stay at the weight level so the architecture matches the public Qwen3-Omni. Extra modules (CTC head, dual encoder, layer-weighted features, MoLE) need a strong case, because every deployment would need patched serving code.
  - Ship prompts, a check set and a model card with the model. Do a licence review of the final data mix before any commercial hand-off.
- **Corrected:** `PIPELINE_PLAN.md` §2, §3.3, §4, §5, §10.5, §10.7; `FINETUNING_GUIDE.md` §3.3, §5 (note), §12; D18 above.
- **Open (`DEPLOYMENT.md` §7):**
  - merge test for HF and Megatron adapters;
  - newer vLLM versions;
  - FP8 quality;
  - hand-off package template;
  - licence review.

### D23. ms-swift as a pinned git submodule, upgraded on purpose (2026-09-29, user question)
- **Question:** vendor ms-swift into the repo, or keep it separate and follow upstream fixes?
- **Decision:** a git submodule `third_party/ms-swift` → `https://github.com/modelscope/ms-swift.git`, pinned at `8ec045582` (upstream `main`, 2026-09-23). This is the commit installed in `swift_megatron_cu128.sif`.
- **Why pin:** `train_omni.py` subclasses ms-swift internals, and several of our fixes depend on this version's exact behaviour:
  - internals used: `SwiftSft._prepare_dataset`, `TrainerFactory.get_trainer_cls`, `MegatronSft`, `MegatronTrainer._prepare_dataloader` / `save_checkpoint`, `args.init_iters`;
  - version-specific fixes: D16, D17, B17, B18.
  An unpinned upgrade could quietly break resume or data order.
- **Why not copy it in:** +120 MB of someone else's code, no history, and pulling upstream fixes becomes a manual diff.
- **Upgrade rule:**
  - branch → move the pin → rebuild the container → re-run the audio-LoRA smoke test and the resume tests → merge;
  - the pin and the container image must always match.
- **Setup:** cloned from the local checkout (`toolkits/ms-swift`, clean, same commit), so nothing was downloaded, then pointed at GitHub. `git submodule absorbgitdirs` moved its `.git` into `.git/modules`.
- **Not changed yet:** `container/build_container.sh` (outside the repo) still builds from `toolkits/ms-swift`. Both are at the same commit.

### D24. enroot image converted from the SIF, same launcher interface (2026-09-29, user: no Apptainer on the new server)
- **Decision:** no rebuild. `container/sif_to_enroot.sh` reuses the SIF's root filesystem and adds `/etc/environment` (the image's Env, from the SIF's JSON config) and `/etc/rc`. The result is `swift_megatron_cu128.sqsh` (12.6 GB, ~1 min to make).
- **Same image, same results:** identical packages mean the Apptainer test results carry over. This was checked, not assumed (job 150477).
- **Launcher:**
  - `run_container.sh` dispatches to `run_container_enroot.sh` when `apptainer` is missing (`CONTAINER_RUNTIME` to force);
  - explicit environment, as with `--cleanenv`;
  - the image is unpacked once per node on local disk;
  - host `/tmp` is bound. enroot's default is a RAM tmpfs on `/tmp`, which would put the 600 GB mosaic cache in memory.
- **Transfer:**
  - `copy_to_h100.sh` takes `IMAGE=sif|sqsh|both`;
  - `relocate_paths.sh` rewrites the bind paths in both launchers.
- **Evidence (job 150477):**
  - 2× H200 visible; the Transformer Engine layer ran on the GPU;
  - `/tmp` on xfs;
  - the audio-LoRA smoke test passed on both trainers, same as under Apptainer: loss 1.030 → 0.723 (HF) / 1.033 → 0.721 (Megatron); memory 61.8 / 34.4 GiB; 24/24 encoder `lora_B` trained.
- **First attempt:** job 150476 failed in the converter. The SIF keeps its config at the top level of the JSON, not under `config` (B21).

### D25. main = NSCC: container files in the repo, PBS scripts, offline pydeps (2026-09-30, user)
- **Decision:** the repo on `main` is the NSCC version. Orion material stays as legacy (`slurm/`, `transfer/`, marked in their READMEs; `pbs/convert_slurm_to_pbs.py` refuses to overwrite `pbs/` without `--force`).
- **Container files are in git** (`container/`: launchers, `.def`, build and convert scripts, `smoke_test.sh`, `README.md`). The big files stay outside in `CONTAINER_HOME=/scratch/users/astar/ares/sailorhb/container` (`.sqsh`, model, `pydeps`, caches); the launchers find them through that variable. The copies in `CONTAINER_HOME` (old launchers, `.bak_*`) are superseded by the repo's.
- **NSCC has no Apptainer**, so the image is the `.sqsh` converted on Orion. Rebuilding needs Apptainer elsewhere (`container/README.md`).
- **pydeps** (`mosaicml-streaming` 0.13 and 4 helpers) is reinstalled in `CONTAINER_HOME/pydeps` from wheels downloaded on the login node: compute nodes cannot reach PyPI and the container gets no proxy. *(Corrected 2026-10-01: compute nodes do reach PyPI, W&B and HF through the site proxy; the container now gets the proxy variables, see D27. Downloading on the login node still works.)*
- **enroot data path:** the image is unpacked per job on the node's `/raid` (NSCC default, ~15 s). An unpack into shared scratch (`CONTAINER_HOME/enroot_data`, ~30 GB) was tried first and is not needed; it can be deleted.
- **Data:** the same MDS tree is at `/data/projects/13003558/zoux/datasets/...`; `mixes/st_v0.yaml` and the tests point there. The launchers bind `/data/projects/13003558`.
- **Launcher change:** `GLOO_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_P2P_LEVEL TORCH_DISTRIBUTED_DEBUG` are forwarded into the container (needed for multi-node).
- **Evidence (NSCC, H100 80 GB):**
  - `train_omni.py sft` 10 steps + resume: 60.3 GiB peak, resume max |loss diff| 0.0069;
  - `train_omni.py megatron` EP=1, 20 iterations + resume + 80 GB cap: 61.5 GiB peak, resume diff 0.0199 (H200 runs: 0.003 to 0.004; not investigated, step 11 matches exactly);
  - `pbs/smoke_audio_lora_2gpu.pbs` via `qsub` (job 214221, 2 GPUs, 12 min): both trainers pass, 24/24 encoder `lora_B` trained, HF 61.7 GiB, Megatron EP=2 34.4 GiB;
  - `tests/test_mosaic_stream.py`: 7/8 on a 1-GPU job; the 2-rank test needs two GPUs on the host (NCCL "Duplicate GPU").
- **Not yet run:** `pbs/train_8gpu_both.pbs` (1 node x 8 GPUs) and `pbs/train_32gpu_4node.pbs` (4 nodes x 8 GPUs, launched with `pbsdsh`) wait for the dedicated queue; the launcher logic was dry-run and the one-node path run for real. Whether InfiniBand works inside the container is unchecked (the 4-node script measures it first).
- **Bug B22:** `smoke_megatron_1gpu` resumed with `--no_save_optim true`, which the D17 guard rejects. Fixed in `slurm/` and `pbs/`.

### D26. Reserved queue R212478 only; multi-node launch through a command file (2026-10-01, user)
- **Decision (user):** every GPU job uses the reserved queue: `#PBS -q R212478`, `#PBS -P 13003558_R4` (12 DGX H100 nodes, 96 GPUs,
  2026-10-01 to 10-31). No other GPU queue (`normal`, `aidev`, `aiq*`, `dedicated`). All `pbs/*.pbs` headers switched.
- The reservation is `place=free`; a job asking for `place=scatter:excl` is refused ("job and reservation have conflicting
  specification Resource_List.place"), so the scripts have no `place=` line. Whole-node chunks (`ngpus=8:ncpus=112`) still land on
  separate nodes.
- **Bug B23 (multi-node):** `pbsdsh` runs its arguments through a shell on each node, so `$RUN` in the phase command was expanded
  there, before `node_run.sh` had sourced `nscc/env.sh`, and came out empty (`python: command not found`, exit 127, job 214958).
  Fix: `run_phase` writes the phase command to `<out>/<phase>.cmd.sh` and the nodes run that file. Note: a phase prints `exit=0`
  even when its node tasks fail; read the `[node_run] ... exit=` lines.
- New `pbs/train_16gpu_2node.pbs` (2 x 8 GPUs). It and the 4-node script take `qsub -v MIX_FILE=mixes/<mix>.yaml` (2-node) and `WANDB=1`.
- **Evidence:** ST mix, global batch 32, 60 steps, resume 30 -> 60, partition check:
  - 2 nodes (job 215070): NCCL busbw 416 GB/s; HF 1.45 s/step, Megatron EP=8 2.2 s/step (14 GiB); resume diff 0.004 / 0.003;
  - 4 nodes (job 215088): NCCL busbw 308 GB/s; HF 0.91 s/step, Megatron 1.64 s/step; resume diff 0.004 / 0.005;
  - partitions disjoint on 16 and 32 ranks; InfiniBand works inside the container.
- While `/data/projects/13003558` was unreadable (2026-10-01, group reset to root for some hours), the 2-node test ran on a
  stand-in mix, `mixes/asr_ta_codeswitch_test.yaml` (jobs 215025, 215052).

### D27. Weights & Biases through each trainer's own callback (2026-10-01, user request)
- **Decision:** `train_omni.py --wandb true` (or `WANDB=1`, as in the user's sortformer scripts). No logging code of our own: it adds
  `wandb` to `--report_to` and lets transformers' `WandbCallback` (rank 0, x axis `train/global_step`) or Megatron-SWIFT's
  callback (last rank, x axis = iteration) log. Neither passes entity / group / tags / run id to `wandb.init`, so `train_omni.py`
  sets `WANDB_*` variables, which `wandb.init` reads.
- **Defaults:** entity `i2r-llm` (user's choice), project `meralion_v4`, run name = basename of `--output_dir`, files in
  `<output_dir>/wandb`, `WANDB_LOG_MODEL=false` (no checkpoint upload). PBS scripts: one group per job, `<job name>_<job id>`.
- **Resume (idea from multimodal_trainer, which keeps the run id in its training state):** each checkpoint gets `omni_wandb.json`,
  written by the rank that holds the run. `--wandb_resume auto` continues that run when the checkpoint is inside `--output_dir`
  (a restart); otherwise a new run starts in the same group, with a "resumed from" note. For Megatron, W&B drops the points
  between the checkpoint and the last logged iteration (it logs against the iteration number).
- **API key:** `WANDB_API_KEY`, else `~/.netrc` (`nscc/env.sh`). Not stored in the repo. (The multimodal_trainer enroot scripts
  hard-code a key; not copied.)
- **Network:** compute nodes reach `api.wandb.ai` only through the site proxy (probe job 215044); the launchers now forward
  `http(s)_proxy` / `no_proxy` and the `WANDB_*` settings into the container.
- **Evidence:** `tests/test_wandb.py` (CPU); jobs 215052, 215070, 215088: all runs finished in W&B with train/eval loss, our
  options, the mix and `omni_effective` in the config. Not exercised live: a restart continuing the same run id.
- **Bug B24:** the HF recipe has `report_to: none`; `none` + `wandb` is refused by transformers (job 215045). `setup_wandb` drops `none`.

### D28. MERaLiON-4 stage 1: multi-task LoRA on 8 nodes (2026-10-02, user brief)
- **Goal (user):** improve ST, spoken QA (CPQA), summarisation, emotion and cultural understanding / reasoning; keep ASR
  (Singapore languages incl. Singlish, SEA languages; very little public English). Several stages allowed.
- **Mix** `mixes/mv4_v0.yaml` from `tools/make_mix_mv4_v0.py` (source: MERaLiON-3 v4.9 + wangq2 Emotional-YTB sets):
  4.74 M samples per epoch, ASR 36% / ST 24% / QA + summary + emotion 40%. **Recipe** `configs/mv4_lora_v0.yaml`: HF DDP,
  LoRA r64 / alpha 64 on LLM attention (48 layers) + audio encoder attention + MLP (32 layers) + projector; LR 1e-4, 500 warmup,
  cosine to 1e-5; batch 256; 18,500 steps. Reasoning and the MERaLiON-3 comparison: `docs/mix_mv4_v0.md`.
- **HF DDP, not Megatron:** faster per step on 4 nodes (0.91 vs 1.64 s at batch 32) and audio-encoder LoRA is tested there;
  62-63 GiB per GPU fits. MoE experts get no LoRA (fused tensors).
- **Run files kept with the results:** `train_omni.py` copies config, mix, PBS script and command into `<run>/omni_run/`.
- **Bugs found:** B25 warmup_steps silently 0 in swift sft (ms-swift + transformers 5.2) -> converted to warmup_ratio;
  B26 mosaic shared memory left by a killed job ("Reused local directory") -> per-node cleanup phase before training;
  B27 `--clean_stale_shm` on all local ranks at once broke the first collective -> not used in multi-rank jobs.
- **Node a2ap-dgx011** lacks `/dev/infiniband/rdma_cm` (enroot mellanox hook fails; jobs 215246, 215261): excluded; the PBS
  script checks the device on every node first. To be reported to NSCC (WORKLOG 2026-10-02).
- Next stages: `docs/MERALION4_ROADMAP.md`.

---

## 4. Problems found and fixed

| # | Problem | Found by | Fix |
|---|---|---|---|
| B1 | PEFT 0.20: a regex `target_modules` like `q_proj\|k_proj\|v_proj` also wraps the audio encoder's attention | LibriSpeech smoke test (09-24) | Explicit full-path target list (`lora_targets_thinker_attn_audio_proj.txt`, 194 modules); check adapter tensor names after training |
| B2 | Container's jiwer 4.x removed `compute_measures` (old WER code) | Metrics check | Fallback to `jiwer.process_words` in `omni_mds/metrics/wer.py` (the only patch to a copied file) |
| B3 | METEOR downloads NLTK data at run time; compute nodes may be offline, and the container home is per-node `/tmp` | Metrics check | Data pre-downloaded to `container/cache/nltk_data`; `run_container.sh` sets `NLTK_DATA` |
| B4 | mosaic `StreamingDataLoader` counts samples from the batch shape: 1 per padding-free batch (`input_ids [1, T]`), so resume would land in the wrong place | Reading mosaic's code, then a test | `OmniStreamingDataLoader` counts `batch_size` per batch; `make_dataloader` checks that the dataset and loader batch sizes match |
| B5 | Dropping unusable samples would give ranks different batch counts and hang DDP | Design review | Unusable samples are *replaced* by a neighbour (tested) |
| B6 | Template encoding took ~2 s/sample on the login node | CPU encode check | Cause: 64 torch threads competing on a busy node. With 1 thread it is 17 ms; dataloader workers already use 1 thread. Keep `OMP_NUM_THREADS` low for main-process encoding |
| B7 | My first ST sample counts were 2× too high | Cross-check with index files | Count each dataset's **top-level** `index.json` only; it already covers the 16 sub-folders |
| B8 | Missing mix paths were reported as 98; the real number is 78 (71 SQA + 7 AC) | Re-count | Corrected in the docs and memory |
| B9 | Instruction dropout: the first port dropped the audio (the old code kept it) | Re-reading the old code | Default restored and parity-tested (D14) |
| B10 | `_meta.sample_id` was a numpy int64 and could not be written as JSON | Two-rank test | Stored as a plain int |
| B11 | mosaic 0.13 declares `transformers<5`, torchvision and cloud SDKs | Wheel metadata | Installed `--no-deps` into `toolkits/pydeps` with only zstd, python-snappy, cramjam, catalogue; checked that the container's numpy/torch/transformers stay in use |
| B12 | mosaic's `clean_stale_shared_memory` removes *every* mosaic shared-memory segment of the user on the node | Reading mosaic's code | Documented: only call it when the job has the node to itself (`--clean_stale_shm`, off by default) |
| B13 | Megatron `init_iters` uses the dataset length as global | Reading ms-swift's code | D16 |
| B14 | Megatron LoRA optimizer resume fails | 1-GPU smoke test | D17 |
| B15 | Megatron default `merge_lora=true`: 66 GB per checkpoint | 1-GPU smoke test | D18 |
| B16 | Megatron `megatron sft` needs `NPROC_PER_NODE` and `CUDA_DEVICE_MAX_CONNECTIONS=1`; its log argument is `--logging_steps` | Container smoke test (09-24) / CLI source | `train_omni.py` sets `CUDA_DEVICE_MAX_CONNECTIONS` and re-launches under torchrun |
| B17 | Saved `adapter_config.json` → `target_modules` matches more modules than were trained. Megatron writes bare names (`q_proj`, `fc1`). swift sft shortens the list (`28.self_attn.k_proj` also matches LLM layer 28). PEFT wraps the extras in zero LoRA at load time | Audio-LoRA smoke test (job 150441) and reading mcore_bridge `gpt_bridge.py` | With encoder LoRA on, `fix_peft_target_modules` rewrites the list after every save to the exact modules in `adapter_model.safetensors` (original kept as `adapter_config.orig.json`). Megatron resume reads only r/alpha/dropout/rslora/bias from that file |
| B18 | ms-swift keeps only the last `--target_modules` flag; with `all-linear` on a multimodal model it builds a regex and drops any extra names | Reading `swift/megatron/utils/utils.py` and the HF argument parsing | Encoder names are inserted into the one existing list; `all-*` and `--target_regex` are refused with a clear message |
| B19 | Megatron-SWIFT refuses `--save_total_limit 1` (`must be >= 2`) | Job 150441 | Test script uses 2 |
| B21 | `sif_to_enroot.sh` looked for `config.Env` in the SIF JSON; the SIF stores the OCI config at the top level | Job 150476 | Accepts both layouts; job 150477 passed |
| B20 | Docs assumed vLLM serves our LoRA adapters; vLLM 0.17.1 has no LoRA support for the Qwen3-Omni thinker | `supports_lora` check in the container (09-29), prompted by the user's deployment question | Merged-model deployment and eval plan (D22, `DEPLOYMENT.md`) |

---

## 5. Evidence: tests and jobs

| When | What | Result |
|---|---|---|
| 09-24 | Container smoke tests (inference, vLLM, swift sft, megatron sft) | All pass (`container/README.md`) |
| 09-24 → 09-28 | **Job 147727**, 8-GPU trainer benchmark | swift 17.0 vs. Megatron 13.5 samples/s; memory 61.8 vs. 14.2 GiB (D5) |
| 09-28 | `tests/test_sea_text.py` (CPU) | Parity 8,392/8,392; dropout parity; augmentation output; AC fix; format checks |
| 09-28 | `tests/test_mosaic_stream.py` (CPU) | 8/8: identical raw samples, `choose` per epoch, fixed validation subset, disjoint 2-rank partitions, exact resume, padding-free counting, replacement not dropping; ~3,400 rows/s |
| 09-28 | `tests/check_encode_cpu.py` | 13.0 audio tokens/s; prompt/audio masked in labels; padding-free batch; 17 ms/encode per thread |
| 09-28 | **Job 149688**, HF 1-GPU smoke | Pass (D15) |
| 09-28 | **Job 149697**, Megatron 1-GPU smoke | Training/eval/save/80 GB cap pass; resume failed (→ D17); 66 GB merged copies (→ D18) |
| 09-28 | **Job 149702**, Megatron resume follow-up | Our data resume works without the optimizer; fully-reshardable format loads but gives NaN |
| 09-28 | **Job 149713**, optimizer debug | Upstream confirmed; optimizer state empty on load |
| 09-28 | **Job 149720**, fix verification | Resume with optimizer within 0.009 (default) / no NaN (fully-reshardable) |
| 09-28 | **Job 149550**, unzip benchmark | Read 4.45 GB/s; unzip 3.3 GB/s; full mix ~9 min per node (D7, D8) |
| 09-29 | **Job 149729**, 8-GPU both paths (ran 09-29 11:47, 20 min) | Pass: resume 0.004 / 0.004 (Megatron with optimizer); 8 × 248 disjoint ids; `eval_iters` 6; memory 61.8 / 16.7 GiB; **steady 14.6 (HF DDP) vs. 13.9 (Megatron EP=8) samples/s**; 4→8 GPU scaling 1.92× / 2.0× (`PIPELINE_PLAN.md` §10.11) |
| 09-29 | **Job 150431**, 4-GPU both paths | Pass: resume 0.004 / 0.003; partitions disjoint; `eval_iters` 6; memory 61.8 / 23.5 GiB (D19) |
| 09-29 | **Job 150441**, audio-LoRA smoke (2 GPUs) | HF passes; Megatron stopped by `--save_total_limit 1` (B19) |
| 09-29 | **Job 150442**, audio-LoRA smoke (2 GPUs) | Pass on both trainers: LoRA only in layers 28–31, 24/24 `lora_B` trained, exact adapter configs (D20) |
| 09-29 | **Job 150477**, enroot image + launcher (2 GPUs) | Pass: GPUs, TE, versions, `/tmp` on disk; audio-LoRA smoke test on both trainers matches Apptainer (D24). Job 150476 failed in the converter (B21) |
| 09-29 | `transfer/relocate_paths.sh` on a local copy | All `.py/.sh/.sbatch/.yaml` rewritten; no `/scratch/` left; `BINDS` updated (D21) |

---

## 6. Where we are (updated 2026-09-29; NSCC status 2026-10-01 in D25-D27 and `WORKLOG.md`)

**Done:**
- the data layer (`omni_mds`);
- the HF training path;
- the Megatron training path, including optimizer resume, checked on 1, 2 and 4 GPUs;
- disk guards;
- the unzip measurements;
- audio-encoder LoRA options (D20);
- the transfer kit for the H100 cluster (D21).

**Next:**
1. ~~Job 149729: 8-GPU throughput~~ **done 09-29:** HF DDP 14.6 vs. Megatron EP=8 13.9 samples/s on `st_v0`; both pass all checks (`PIPELINE_PLAN.md` §10.11).
2. Copy to the H100 cluster (`transfer/TRANSFER_TO_H100.md`) and run `container/smoke_test.sh env` there.
3. Megatron re-run with bigger batches or packing, for a fair trainer choice (D5).
4. Eval script: transformers + PEFT for in-training checks, merged + vLLM for final numbers (D22); the old metrics, per-task/language scores (phase 5), then a zero-shot baseline.
5. Full-mix YAML and the data inventory (hours per task × language); fix or drop the 78 missing paths.
6. ~~A 2-node run (multi-node mosaic partitions; InfiniBand).~~ **done 10-01:** 2 and 4 nodes pass (D26).
7. ~~ASPIRE2A+: profile files (`profiles/h100_a2ap.env`), PBS job headers.~~ **done 09-30/10-01:** `nscc/env.sh`, `pbs/` on R212478 (D25, D26).
8. Test `swift export --merge_lora` on an HF and a Megatron adapter (with audio-encoder LoRA) and serve the result in vLLM; compare with PEFT outputs (`DEPLOYMENT.md` §7).
9. Licence review of the final data mix before any commercial hand-off (`DEPLOYMENT.md` §6).
10. Point `container/build_container.sh` at `third_party/ms-swift`, so the image is always built from the pinned submodule (D23).

**Open decisions:**
- target text convention (speaker tags, `#entity#` hashtags, also in SDS targets);
- system prompt;
- final trainer for LoRA runs;
- whether to post the upstream issue.

---

## 7. What was created or changed

**New in `qwen3_omni_speechllm/`:**
- **Package `omni_mds/`:**
  - `mds_io.py`, `mix.py`, `sea_text.py`, `mosaic_stream.py`, `audio/augment.py`;
  - copied from the old trainer: `instructions_lib/`, `text_normalizers/`, `metrics/`, `audio/data_augmentation.py`, `language_mapping.py`, `prompt_utils_orig.py`;
  - `VENDORED.md`.
- **Scripts:**
  - `train_omni.py`;
  - `mixes/st_v0.yaml`;
  - `slurm/`: `smoke_1gpu`, `smoke_megatron_1gpu`, `smoke_megatron_resume`, `debug_megatron_optim_resume`, `verify_megatron_optim_resume`, `train_8gpu_both`;
  - `tests/`: `test_sea_text.py`, `test_mosaic_stream.py`, `check_encode_cpu.py`, `summarize_smoke.py`, `check_partitions.py`;
  - `tools/vendored/`: the old MDS helper scripts.
- **Docs:** `PIPELINE_PLAN.md`, `DECISION_LOG.md`, `docs/upstream_issue_megatron_resume.md`; `MDS_DATA_PIPELINE.md` §0 and §0.5 added; counts corrected.
- **Added 2026-09-29:**
  - `train_omni.py`: `--audio_lora_layers`, `--audio_lora_modules`, `add_audio_lora_targets`, `fix_peft_target_modules`, `argv_value`;
  - `tests/summarize_smoke.py`: reports encoder LoRA layers and modules, trained `lora_B` count and `adapter_config` targets;
  - `slurm/train_4gpu_both.sbatch`, `slurm/smoke_audio_lora_2gpu.sbatch`;
  - `transfer/`: `TRANSFER_TO_H100.md`, `copy_to_h100.sh`, `relocate_paths.sh`, `requirements.txt`, `requirements_pydeps.txt`, `requirements_container_freeze.txt`;
  - docs: `PIPELINE_PLAN.md` §10.6–10.8, `FINETUNING_GUIDE.md` §5.1, `WORKLOG.md` 2026-09-29.
  - git repo (`git@github.com:HardikSailor/qwen3_omni_speechllm.git`): `README.md`, `.gitignore`; submodule `third_party/ms-swift` @ 8ec0455 (D23);
  - `DEPLOYMENT.md` (D22).

**Outside `qwen3_omni_speechllm/`:**
- `toolkits/pydeps/`: mosaicml-streaming 0.13.0 + zstd, python-snappy, cramjam, catalogue (`--no-deps`).
- `container/run_container.sh`:
  - forwards `PYTHONPATH SWIFT_AUDIO_LOAD_BACKEND OMNI_GPU_MEM_GB OMNI_MDS_CACHE OMNI_DEBUG_OPTIM_LOAD OMNI_LOG_SAMPLE_IDS`;
  - sets `NLTK_DATA`.
- `container/cache/nltk_data/`: wordnet, omw-1.4, punkt_tab.
- `container/unzip_bench/`: unzip benchmark script and logs.
- Not modified: ms-swift (`toolkits/ms-swift`), `multimodal_trainer`, `zoux`'s data, `nemo_env`.

**Deleted:** three Megatron `checkpoint-*-merged` folders (~200 GB, our smoke-test outputs only).
