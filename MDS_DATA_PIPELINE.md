# Using our Mosaic (MDS) speech datasets with ms-swift / Megatron-SWIFT for Qwen3-Omni

Written 2026-09-24. Sources studied:
- `toolkits/multimodal_trainer/modules`: the old MERaLiON pipeline (datasets, collators, normalisers, metrics);
- `toolkits/ms-swift` @ 8ec0455: the same commit as inside the container;
- the MDS shards themselves (decoded, see §1).

Measured numbers come from node DDS9B74 inside `container/swift_megatron_cu128.sif`. Code sketches in §5 have not been run yet.

---

## TL;DR

**Recommendation: write a thin, map-style MDS adapter (option D in §3).** Plug it into the unmodified `swift sft` / `megatron sft` pipelines with a ~20-line entry script. Do not convert the corpus to JSONL, parquet or Arrow. Do not use ms-swift's `--streaming` mode.

- **Keep from the old code:** the MDS data and the dataset YAMLs (`choose`, `repeat`, `weight`), the text normalisers, the multilingual instruction library, audio augmentation, and the metrics (WER/MER, BLEU, METEOR).
- **Drop from the old code:** the old collators, trainer, FSDP code and `<SpeechHere>` prompt splitting. They are tied to the MERaLiON encoder+adapter+LLM model. ms-swift's Qwen3-Omni template already does tokenisation, audio features, padding and padding-free for this model.
- **Change from the old code:** the old pipeline copies every shard into a 600 GB-per-node Mosaic cache. Instead, decompress the shards once into a random-access cache on each node's local NVMe (`/raid` = `/tmp`, 28 TB free per node), or once to `/scratch` if quota allows. After that, reads are plain `pread` calls.
- **Container:** yes. `swift_megatron_cu128.sif` already has everything the adapter needs (zstandard, soundfile, pyav, librosa, jiwer, regex, more_itertools, scipy). No rebuild is needed. You only need to forward `PYTHONPATH` and one environment variable in `run_container.sh` (§6).

Why not the obvious alternatives:
- A full export to JSONL + audio files would mean about 82 M small files and another 2 TB on a `/scratch` that is 86% full.
- ms-swift's `--streaming` has rank 0 read and pickle-scatter every batch to all ranks (`DataLoaderDispatcher`). That does not scale to 8–100 GPUs of audio.
- The map-style path is the one ms-swift and Megatron-SWIFT are optimised for. Each rank loads its own samples in its own workers, resume is deterministic, and it supports `group_by_length` and padding-free.

---

## 0. Old vs. new: how datasets are handled (overview)

Added 2026-09-28. This section summarises the design stage by stage. Details and evidence are in §1–§8. **Status:** design only; no adapter code has been written or run yet.

In one sentence: the **data stays where and how it is** (same MDS shards, same mix YAMLs). What changes is **how it is read** and **where the per-sample text processing runs**. Everything that turns text and audio into model inputs moves to ms-swift's Qwen3-Omni template.

### 0.1 Side by side

```
            multimodal_trainer (old)                    Qwen3-Omni plan (new)
            ─────────────────────────                   ─────────────────────
Data        MDS shards (.mds.zstd) + mix YAML           same shards, same YAML format
              │                                           │
Access      mosaic StreamingDataset                     one-time: unzip shards to node-local NVMe
            copies shards into a 600 GB/node cache       + an index file per dataset
            and reads them sequentially                  (shard, row, task, lang, duration)
              │                                           │
Mixing      mosaic Streams with choose/repeat           same choose/repeat, re-drawn each epoch
              │                                           │ from a seed → one list of sample ids
Per-sample  InstructDataCollator (per batch):            row_fn (per sample), code ported from the old collator:
text logic  normalise, pick prompt, dropout,             normalise, multilingual prompt, dropout
            Whisper features, split at <SpeechHere>      → {"messages":[user "<audio>"+prompt,
              │                                                          assistant answer],
              │                                              "audios":[raw bytes]}
              │                                           │
Model       custom collator → custom Trainer/FSDP →     ms-swift Qwen3-Omni template (audio tokens,
inputs      MERaLiON encoder+adapter+LLM                 chat format, labels, padding-free) →
                                                         swift sft  or  megatron sft (both unchanged)
```

| Stage | Old (`multimodal_trainer`) | New (Qwen3-Omni) | Change |
|---|---|---|---|
| Storage | MDS shards + dataset YAML | Same shards, same YAML format | None |
| Reading | mosaic `StreamingDataset`, sequential, 600 GB/node cache | Shards unzipped once to node-local NVMe; one `pread` per sample | Replaced |
| Index | None | Parquet index per dataset (shard, row, task, language, duration) | New |
| Mixing | mosaic Streams, `choose`/`repeat` | Same semantics, re-drawn per epoch from a seed | Re-implemented, same behaviour |
| Text logic | `InstructDataCollator`, per batch | `row_fn`, per sample | Ported, with deliberate format changes |
| Model inputs | Custom collator, Whisper features, `<SpeechHere>` split | ms-swift Qwen3-Omni template | Replaced by ms-swift |
| Trainer | Custom Trainer + FSDP | `swift sft` / `megatron sft`, unmodified | Replaced |
| Eval | Generation per val set inside training | Loss in training; separate vLLM eval with the old metrics | Moved out of the training loop |

### 0.2 Stage by stage

**1. Storage: no change.**
- We keep the MDS shards and the YAMLs with `name`, `path`, `task`, `choose`, `repeat` and `weight`.
- We don't convert anything to JSONL or parquet. That would mean about 82 M small files or another 2 TB+ on a `/scratch` that is almost full (§3).

**2. Reading: sequential streaming → random access** (§4.1, §4.5).
- Old: mosaic streams shards into a per-node cache and reads them in order.
- New: ms-swift's samplers need to fetch **any sample by its index**. That is what makes these work:
  - each GPU loading its own share;
  - exact resume after a crash;
  - batching clips of similar length together;
  - Megatron's samplers, which must give TP/CP peers the same batch.
- Our shards are zstd-compressed, and you can't jump to the middle of a zstd file. So at job start each node unzips the shards once to its local NVMe (`/tmp`, 28 TB free; about 2 TB for the full mix, done in minutes). After that, a small reader of about 30 lines pulls one sample with a single `pread`.
- `mosaicml-streaming` is no longer needed.

**3. A one-time index per dataset (new)** (§4, §4.4).
- It stores shard, row, task, language and audio duration for each sample, and takes about 1 h over the whole mix.
- It gives us:
  - the length of every sample before training (audio is 13 tokens/s), for batching by length, packing and dropping over-long samples;
  - a fix for the 34 shards that have no `language` column;
  - the hours-per-task-and-language table we need for the data inventory anyway.

**4. Mixing: same behaviour** (§4.2).
- `choose: N` / `repeat: r` are applied per dataset and re-drawn each epoch from a seed, as before.
- `weight` stays as the weight for the combined eval score.

**5. Per-sample text processing: ported, with a few deliberate changes** (§4.3).
- **Ported as is from `InstructDataCollator`:**
  - answer clean-up (`_standardize_response`): fillers, Chinese/Thai handling, speaker tags, hashtags;
  - random prompts in 13 languages from `instructions_lib`;
  - instruction dropout.
- **Changed:**
  - the MERaLiON system prompt and the `<SpeechHere>` / `<TextHere>` layout are replaced by Qwen's own format: audio first, then the instruction;
  - the old code picked prompts per batch; the new code does it per sample.
- **Rule:** `context_text` never goes into the prompt, because for ST and ER it is the transcript and would give away the answer.
- **Decision needed:** one convention for speaker tags and `#entity#` hashtags in the targets. The model copies the style of its targets: training on all-caps LibriSpeech targets made the WER worse.

**6. Turning text and audio into model inputs: ms-swift does it** (§2.2).
- All of this is done by ms-swift's Qwen3-Omni template, which we have already tested:
  - Whisper feature extraction;
  - tokenising and building labels;
  - padding;
  - inserting the audio tokens.
- The raw audio bytes from MDS go straight in, with no temporary files. Encoding happens inside the dataloader workers (`--lazy_tokenize true`).
- **The data code is the same for `swift sft` and `megatron sft`.** One small override (`_get_dataset`) plugs our dataset into both, so the trainer choice doesn't affect the data work.

**7. Evaluation: moved out of the training loop** (§4.6).
- Old: the trainer ran generation on every validation set during training.
- New:
  - during training we track only the loss, on small validation subsets per task and language (e.g. 512 samples each);
  - each saved checkpoint gets a separate vLLM script over the same MDS validation sets, scored with the **old** normalisers and WER/MER/BLEU/METEOR code and weighted by the YAML `weight`;
  - so the numbers stay directly comparable with MERaLiON results.

### 0.3 Open data issues

- **78 of the 195 training paths** (and 20 of 68 validation paths) in the v4.3 YAML are unreadable: 71 `wonghmj/data/SQA/mds_qaed_cleaned_reasoning/yt_sea_hf_v0.2_*` sets and 7 `wonghmj/data/AC/mds_summarized_caption_split_cleaned/yt_sea_hf_v0.2_*` sets. We need access to them or a new location. (Counts corrected 2026-09-28; earlier text said 98.)
- **Two deliberate losses from the old pipeline:**
  - audio augmentation has to be rewritten: it mixed noise from other clips in the same batch, and we now work one sample at a time;
  - Megatron packing needs a small extra class that reads lengths from our index. Until then we use padding-free batching.

### 0.4 Build order

About 1–2 days to a working smoke test (full plan in §7):
1. The reader and the index, checked against one WAV dataset (IMDA) and one Opus dataset (CoVoST2).
2. `row_fn`, with its output compared against the old collator on real samples.
3. A mix dataset plus the ms-swift entry script, then a 1-GPU LoRA smoke test on a 3-dataset mix.
4. Unzipping at job start, plus an 8-GPU run to check data-loading speed and exact resume.

### 0.5 Streaming doesn't avoid unzipping (alternative: mosaic streaming plugged into ms-swift)

> **Measured 2026-09-28 (job 149550, node 474Y574, 80 shards ≈ 36 GB per test, a different shard set each time):** reading from `/scratch` (WekaFS) with 32 in parallel: **4.45 GB/s**. `zstd -d` to node-local NVMe (`/tmp` = md0 over 2 × 3.84 TB NVMe): 8 in parallel 2.39 GB/s, **32 in parallel 3.32 GB/s (3.69 GB/s written)**, 64 in parallel 2.96 GB/s. So the full 1.81 TB mix unzips in **~9 min per node**; reading alone is ~7 min. For mosaic streaming this cost is spread over training. Training reads a few MB/s (8 GPUs × ~2 samples/s × ~0.2 MB), so re-unzipping after cache evictions is negligible.
>
> **Decision (2026-09-28): option C, mosaic streaming plugged into ms-swift, was chosen.** Implemented in `omni_mds/mosaic_stream.py`; results in `PIPELINE_PLAN.md` §10.2. The map-style design (§3 D, §4) is kept below for reference.

Added 2026-09-28. The question: if the index + unzip step feels too heavy, can we keep the old trainer's mosaic `StreamingDataset` for data and use ms-swift / Megatron-SWIFT only for training ("best of both worlds")? **Yes.** This is option C in §3, refined below. It is a sound choice, but it should be a **data plug-in into the stock ms-swift trainers**, not a new training pipeline.

**Streaming doesn't avoid unzipping.** Mosaic cannot read `.mds.zstd` in place either. `StreamingDataset` copies each shard into its local cache and unzips it there. That is why the old trainer needed a 600 GB-per-node cache.

| | Map-style adapter (option D, recommended in §4) | Mosaic streaming plug-in (option C) |
|---|---|---|
| When unzipping happens | Once, up front: to node-local NVMe per node, or once to `/scratch` | Gradually during training, in the background, shard by shard |
| Repeats | Never, if unzipped to `/scratch` | Every job on every node, and again each epoch: the mix (~2 TB unzipped) is bigger than the 600 GB cache |
| Delay before step 0 | ~9 min per node for the full mix (measured, job 149550) | Almost none |
| Permanent extra disk | ~2.0 TB (on `/scratch`), or none with per-node staging | None (transient cache on `/tmp`) |
| One-time index build | ~1 h (lengths, languages, hours table) | None |
| Shuffle | Global, over the whole mix | Within blocks of shards (mosaic `py1e`); the old trainer lived with this |
| `group_by_length` | Yes (lengths from the index) | No; padding-free / packing mostly compensate |
| Packing | Megatron `--packing` needs a small `PackingDataset` subclass fed from the index | On the fly via ms-swift's `IterablePackingDataset` (`swift/pipelines/train/sft.py:142`) |
| Mid-epoch resume | ms-swift's own (skip batches / Megatron `consumed_samples`) | Must wire `StreamingDataLoader.state_dict()` into checkpoints ourselves |
| Change GPU count mid-run | Index-based order changes with world size | Deterministic via `num_canonical_nodes` |
| Megatron layouts | Any (uses Megatron's data-parallel sampler) | Fine for EP-only (TP=PP=CP=1, every rank is a DP rank). TP/CP > 1 needs mosaic told the DP rank/size instead of the global rank |
| `choose` / `repeat` / YAMLs | Re-implemented, same behaviour | Native in mosaic `Stream`; old `init_train_stream()` reused as is |
| New dependency | None | `mosaicml-streaming`, not in the container. Install via `pip --target` and check it works with torch 2.10 / Python 3.11 (the old pin was 0.7.6) |
| New code | ~350 lines | ~300 lines |

**Where the plug-in goes** (checked in the ms-swift source at commit 8ec0455):
- HF trainer: `swift/trainers/mixin.py:1398` `get_train_dataloader()`. For a dataset without `__len__`, it wraps the DataLoader in `DataLoaderDispatcher` (line 1452): rank 0 reads every batch and scatters it. Override this to give each rank a plain DataLoader over its own mosaic partition, with no dispatcher.
- Megatron-SWIFT: `swift/megatron/trainers/base.py:1045` `_prepare_dataloader()`. In streaming mode it also uses a dispatcher, over the data-parallel group (`swift/megatron/trainers/utils.py:330`). Same override.
- Everything else stays stock: the Qwen3-Omni template (`template.encode` + `data_collator`, padding-free), LoRA, expert parallelism, checkpointing and PEFT export.

**Why not ms-swift's built-in `--streaming true` as-is:** rank 0 does all audio decoding and feature extraction for every GPU, and Megatron also limits it to one dataloader worker. On one node at ~17 samples/s (job 147727) that might be tolerable. Across nodes it becomes the bottleneck.

**Why not write our own training loop from ms-swift parts:** it would mean re-owning the parts that already work and are tested: checkpointing/resume, LoRA saving and export, DeepSpeed and gradient accumulation, Megatron's expert parallelism and training loop. The data side is the only part that needs to change, and both trainers have one clean method to override.

**Decision rule.** Both options share the ported text logic (`row_fn`), the Qwen3-Omni format, the trainers and the eval script. They differ only in the reader/sampler layer, so:
1. Build `row_fn` first; it is needed either way.
2. If a permanent unzipped copy (~2.0 TB under our own `/scratch/prj0000000234/sailorhb/`) is allowed, use **map-style** (option D). Otherwise use the **mosaic plug-in** (option C). The data belongs to `zoux` (`/scratch/prj0000000234/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/`), so ask before copying it.
3. In neither case write our own training loop.

**Related: Hugging Face versions of the datasets are not a shortcut.**
- About half the mix by size (IMDA NSC parts 1–6, vendor sets such as King-ASR, Datatang, Willdata, MagicHub, i2r Tamil) is not on Hugging Face.
- The public sets there are not the same data: our shards were already cut to ≤30 s, normalised and turned into instruction, ST, SQA and SDS pairs.
- `datasets.load_dataset` downloads and then writes a full Arrow copy, a one-time conversion like unzipping, but slower. Its streaming mode is sequential, like mosaic.
- Hugging Face is still useful for **adding** public sets not in the mix (e.g. FLEURS languages).

**Why zstd costs us more than it saves:** zstd saves only 2% on a People's Speech shard (Opus audio, 515.6 vs 524.3 MB) and 22% on an IMDA shard (WAV, 410.5 vs 524.3 MB). But it rules out random access until each shard is unzipped.

---

## 1. What our data actually is (verified)

**Layout.** Each dataset directory holds sub-directories `0/ … 15/`. Each sub-directory has an MDS v2 `index.json` and roughly 500 MB `shard.NNNNN.mds.zstd` files. The shards are **zstd-compressed only**: no raw `.mds` files exist. zstd frames cannot be seeked, so random access needs decompression first.

**Schema.** It is identical across all 6,468 shards that the current mix references:

| column | type | contents |
|---|---|---|
| `context_audio` | bytes | The input audio: WAV PCM16 16 kHz (IMDA etc.) or OGG/Opus 16 kHz (CoVoST2, SDS …) |
| `instruction_text` | str | The dataset's instruction, e.g. "Please transcribe.", "Please translate the given speech to Indonesian" |
| `answer_text` | str | The target. It contains `<Speaker1>:` speaker tags and `#char-kway-teow#`-style entity hashtags that the old normaliser post-processes |
| `context_text` | str | **The transcript of the audio** for ST and PQA/ER. The old collator never feeds it to the model. It must not go into the prompt (it would leak the answer), but it is useful for CoT/cascade targets (guide §6.3) |
| `language` | str | Often empty for ST/SDS/PQA. **Missing entirely in 34 shards** |
| `task` | str | ASR, ST, SDS, PQA, CPQA, AC, AQA |
| `answer_audio`, `instruction_audio` | bytes | Empty in every sample checked |

**Size of the current mix** (`multimodal_trainer/config/dataset/meralion2_ctm_2611_with_new_data_with_sampling_v4.3.yaml`):

| | datasets | samples | zstd on disk | decompressed |
|---|---|---|---|---|
| train (accessible) | 117 of 195 | 81.9 M | 1.81 TB | 2.01 TB |
| validation (accessible) | 48 of 68 | 0.32 M | 10.6 GB | 12.4 GB |

- **78 of the 195 train paths do not exist** or are not readable (recounted 2026-09-28; earlier text said 98). 71 are under `/scratch/prj0000000234/wonghmj/data/SQA/mds_qaed_cleaned_reasoning/yt_sea_hf_v0.2_*` (the SQA/reasoning data) and 7 under `/scratch/prj0000000234/wonghmj/data/AC/mds_summarized_caption_split_cleaned/yt_sea_hf_v0.2_*`. Ask for access or a new location before relying on this YAML. Sample counts: always read each dataset's top-level `index.json` only; it already covers the `0/ … 15/` sub-folders.
- 81 datasets use `choose:` (subsampling per epoch). None use `repeat:`.
- zstd saves only about 10% here, because PCM WAV compresses poorly and Opus is already compressed. Storing the shards uncompressed costs little.

**Throughput** (one core, one shard; the file may have been in page cache):

| step | WAV shard (IMDA, avg 23.5 s clips) | Opus shard (CoVoST2, avg 5.9 s clips) |
|---|---|---|
| read from `/scratch` | ~1.3 GB/s | ~1.5 GB/s |
| zstd decompress | ~0.7 GB/s | ~2.4 GB/s |
| decode audio (soundfile) | ~830 clips/s | ~120 clips/s |

For comparison, training consumes about 2–3 samples/s per GPU (smoke test: 1.7 s/step, batch 1 × grad-accum 4). So once shards are decompressed, one or two dataloader workers per GPU are plenty. Decompressing a whole 500 MB shard **per sample** is not acceptable: that is what forces the cache/staging step.

## 2. What each side does

### 2.1 Old pipeline (`multimodal_trainer`)

```
YAML (name, path, task, choose/repeat, weight)
  → init_train_stream(): one mosaic Stream per dataset, local cache <mds_cache_dir>/train/<node>-<name>
  → StreamingDataset(streams, shuffle=True, num_canonical_nodes=world/8, cache_limit=600gb, predownload)
  → MultimodalStreamingDataLoader(collate_fn=InstructDataCollator)
       per batch: decode bytes → normalise answer (ASR) → random/multilingual prompt → optional text-only swap
                  → Whisper feature extractor → chat template split at <SpeechHere> → left/right token tensors
  → custom Trainer / FSDP / MERaLiON model (speech_contexts_features, text_prompts_left/right, …)
```

Everything from the feature extractor onwards belongs to the old model and cannot be reused with Qwen3-Omni. The reusable parts are the reading, the text logic and the metrics.

| module | reuse? | note |
|---|---|---|
| `datasets/datasets.py`, `multimodal_streaming_dataloader.py` | concept only | Replaced by the map-style reader. Keep the YAML semantics (`choose`, `repeat`, `weight`) |
| `data_collators/data_collator.py` `_convert_mds_*` | yes (trivial) | Decoding bytes → audio |
| `instruct_collator.py` `_standardize_response`, `get_prompt`, instruction dropout | **yes, port** | Becomes a per-sample `row_fn` (§4.3) |
| `prompt_utils.py`, `instructions_lib/*` (13 languages) | **yes, as is** | Pure Python |
| `text_normalizers/wer_text_normalizer` | **yes, as is** | Needs jiwer, regex and more_itertools, which are all in the container |
| `data_augmentation.py` | yes, adapt | It is batch-based: crowd noise is mixed from other clips in the batch. Make it per-sample with a small per-worker noise reservoir |
| `language_mapping.py` | optional | Only needed if we add language soft tokens (guide §5.7) |
| `metrics/` (wer, bleu, meteor) | **yes** | For generation-based eval (§4.6) |
| collators' tokenisation/left-right split, `batch_processor.py`, `distributed/`, `trainer.py`, `models/` | no | ms-swift's template, trainer and DeepSpeed/Megatron replace them |

### 2.2 ms-swift / Megatron-SWIFT (checked in source)

- **Row format** expected by the Qwen3-Omni template: `{"messages": [{"role": "user", "content": "<audio>…"}, {"role": "assistant", "content": "…"}], "audios": [x]}`. Here `x` may be a path, URL, base64 string **or raw `bytes`**. `vision_utils.load_file` wraps bytes in a `BytesIO`, then librosa or soundfile+pyav decodes them and resamples to 16 kHz. So MDS audio bytes can go straight through, with no temporary files. `SWIFT_AUDIO_LOAD_BACKEND=soundfile_pyav` is the faster decoder.
- **`lazy_tokenize`**: `SwiftSft._post_process_datasets` wraps any object with `__len__` and `__getitem__` in `LazyLLMDataset(dataset, template.encode)`. That object does not have to be an HF `Dataset`. It encodes on the fly inside the dataloader workers and skips bad samples.
  - Without `lazy_tokenize`, ms-swift runs `AddLengthPreprocessor` over the whole dataset first. For 82 M audio rows that means decoding every clip before step 0, and writing another Arrow copy. Always set `--lazy_tokenize true`.
- **Map-style dataloading:**
  - HF trainer: `BatchSamplerShard` gives each rank its own index shard. It supports `group_by_length` (it reads `dataset['lengths']`) and resumes by skipping batch indices, not data.
  - Megatron: `MegatronPretrainingRandomSampler` does the same over the data-parallel group and resumes via `consumed_samples`.
- **`--streaming`**: `DataLoaderDispatcher` / `MegatronDataLoaderDispatcher`. Rank 0 of the DP group pulls `world_size` batches and `scatter_object_list`s them (pickled, including audio features) to every rank. Megatron also forces `dataloader_num_workers=1`. Only a shuffle buffer is available, `group_by_length` is not, and `--train_iters` is required. This is the wrong mode for us.
- **Hook point:** `MegatronSft` subclasses `SwiftSft`. A single override of `_get_dataset()` feeds both `swift sft` and `megatron sft`. That method is only called when `args.dataset` is non-empty, so we pass the mix YAML as `--dataset`.
- Qwen3-Omni **supports padding-free** (it inherits `support_padding_free` from `Qwen2VLTemplate`). Audio token count = `_get_feat_extract_output_lengths`: **13 tokens per second** of 16 kHz audio (a 30 s clip is about 390 tokens). The dataset can therefore give an exact length from the clip duration without encoding anything.

## 3. Options

| | A. Export to ms-swift JSONL + audio files | B. Export to HF parquet/Arrow (bytes column) | C. mosaic `StreamingDataset` behind a custom dataloader | **D. Map-style MDS adapter (recommended)** |
|---|---|---|---|---|
| New code | ~100 lines converter | ~100 lines converter | ~300 lines: dataset, override `get_train_dataloader` + Megatron `_prepare_dataloader`, stateful resume | ~350 lines: index, reader, row_fn, entry script |
| Extra disk | +2 TB on `/scratch` and **~82 M files** (inode pressure) | +2 TB, and **another +2 TB** when ms-swift's `.map()` preprocess writes its cache | local cache (like today's 600 GB/node) | +2 TB local NVMe per node (28 TB free), **or** +2 TB once on `/scratch` |
| Up-front time | hours to days (82 M writes) | hours, plus ms-swift preprocessing over 82 M rows | none | index build ~1 h (parallel, once); staging ~10–15 min per new node |
| Multi-GPU loading | good (map-style) | good if non-streaming | good (each rank streams its own partition) | good (map-style, per-rank workers) |
| Megatron TP/PP/CP | fine | fine | **wrong by default**: mosaic partitions by global rank, but TP/CP peers must see the same batch | fine: uses Megatron's DP sampler |
| `group_by_length`, padding-free, exact resume | yes | yes | no `group_by_length`; resume needs mosaic `state_dict` wiring | yes: lengths come from the index; resume uses ms-swift's own mechanism |
| Keeps `choose`/`repeat`/prompt randomisation per epoch | frozen at export time | frozen at export time | yes | yes (re-drawn per epoch from the seed) |
| New dependency | none | none | `mosaicml-streaming` (not in container) | none |
| Verdict | only for **small pilots** (we already do this for LibriSpeech via `make_swift_asr_data.py`) | no | fallback if data moves to S3/object store | **yes** |

## 4. Recommended design (D)

```
mix.yaml (same format as old dataset YAMLs; + optional per-dataset overrides: language, prompt_type, max_seconds)
   │
   ├─ build_mds_index.py  (offline, once per dataset, parallel over shards)
   │     → <index_dir>/<dataset>.parquet: shard_id, row, task, language, duration_s, answer_chars
   │       (+ shard table: zstd path, raw path, sample count, column layout)
   │
   ├─ stage_mds.py  (job prologue, 1 task/node; skipped if cache already valid)
   │     zstd -d every referenced shard → /tmp/$USER/mds_cache/<sha1(path)>/shard.NNNNN.mds (node-local NVMe)
   │
   └─ train_omni_mds.py  = SwiftSft / MegatronSft with _get_dataset() overridden
         MDSMixDataset(mix.yaml, epoch_seed)  ── __getitem__(i) ─→ row_fn(raw MDS sample)
             • pread the sample from the uncompressed shard (offset table from the shard header)
             • SEA text logic ported from InstructDataCollator (normalise, prompts, dropout)
             • optional waveform augmentation → in-memory WAV bytes
             → {"messages":[user "<audio>"+prompt, assistant answer], "audios":[bytes]}
         ['lengths'] → text tokens estimate + 13 × duration_s   (group_by_length / packing planning)
         ↓  (unchanged ms-swift from here)
         LazyLLMDataset(template.encode) → BatchSamplerShard | MegatronPretrainingRandomSampler
         → template.data_collator (padding / padding-free) → Qwen3-Omni thinker
```

### 4.1 Reader (no mosaic dependency)

MDS v2 shard layout: `uint32 n` | `uint32 offsets[n+1]` | samples. Each sample is `uint32 sizes[k]` for its variable-width columns, followed by the column bytes. I checked this against real shards (both WAV and Opus) when decoding the samples in §1. Opening a decompressed shard means reading `4·(n+2)` bytes once and caching the offsets per worker. Each sample is then one `os.pread` (0.02–1 MB). With 2 TB of RAM per node, the page cache will hold much of the hot set.

Keep column order per shard from `index.json`, because the 34 shards without `language` have different column indices.

### 4.2 Mixing semantics (same behaviour as the old YAMLs)

For each "virtual epoch" `e`, per dataset:
- `choose: N` → draw N indices, seeded by `(seed, e, name)`. Without replacement if N ≤ size, otherwise full copies plus a remainder.
- `repeat: r` → equivalent to `choose = round(r · size)`.

Concatenate the draws into an int64 array of `(dataset_id, shard_id, row)` of length L. ms-swift's sampler then shuffles 0..L-1 globally. For multiple epochs, either precompute `num_train_epochs` virtual epochs back-to-back (simplest, so the sampler sees one long dataset), or rebuild the array in the `set_epoch` hook.

`weight` in the old YAML was only used for eval weighting. Keep it for the per-task eval summary.

### 4.3 `row_fn`: porting the SEA text logic and changing the prompt format

Port these parts of `InstructDataCollator` as they are:
- `_standardize_response`: fillers, Chinese/Thai standardisation, the gigaspeech2-id lower-casing, hashtag and apostrophe handling, speaker tags;
- `get_asr_prompt` / `get_ac_prompt` with `random_multilingual` (the 13-language `instructions_lib`), the Vietnamese format-hint augmentation, and `instruction_dropout_rate`.

Change these deliberately:

| old | new for Qwen3-Omni | why |
|---|---|---|
| System prompt "You are … MERaLiON-AudioLLM …" | **none** (template `default_system=None`), or one fixed short prompt used in both train and eval | Stay close to Qwen3-Omni's own instruction distribution |
| `"Given the following audio context: <SpeechHere>\n\nText instruction: <TextHere>"` and the left/right split | user content = `"<audio>" + instruction` (audio first, the pattern in Qwen's own examples) | The template inserts `<\|audio_start\|><\|audio_pad\|>×N<\|audio_end\|>` itself |
| Instruction dropout swaps prompt/response inside the collator | Keep it as it is (a text-only row without `audios` is valid), **or better**, make text-only instructions their own stream with a `choose` | Clearer accounting of the mix |
| `language` from the column | Column, falling back to a per-dataset override in the YAML, falling back to parsing the dataset name (`gigaspeech2_th_…`) | 34 shards have no column, and many rows are empty |
| `context_text` ignored | Still ignored as input. Optionally use it as a CoT target: "transcript → translation" | Leakage (see §1) |
| Targets: speaker tags `<Speaker1>:` and `#entity#` | Decide one convention before training (guide §4.3/§4.4). The model copies target style (see the LibriSpeech smoke test in the LoRA memory: WER got worse with all-caps targets) | |

Return audio as the original **bytes**. When augmentation is on, decode, augment, and re-encode with `soundfile.write(BytesIO, …, format="WAV", subtype="PCM_16")`, which takes under 1 ms per clip.

### 4.4 Lengths, batching, long audio

- `lengths[i] = prompt_tokens_est + 13·duration_s + answer_chars/3` (rough). This is good enough for `group_by_length` and for skipping samples longer than `max_length` before any decoding happens.
  - Duration comes from the WAV header, or from `soundfile.info` on the bytes for Opus. Compute it once in the index build: one pass over 1.8 TB at about 1 GB/s/core, so roughly 1 h on 32 cores.
- Prefer `--padding_free true` (supported by the template), plus `group_by_length` on the HF trainer.
  - For Megatron, `--packing true` needs lengths for every sample up front. ms-swift's `PackingDataset` would encode everything first, so either skip packing or feed our pre-computed lengths (a small subclass; do it later).
- Most sets are capped at 30 s (`_30_` in the names). Qwen3-Omni itself handles longer audio. `--max_length 2048` covers 30 s of audio (~390 tokens) plus long SDS answers.

### 4.5 Staging / cache

- **Default:** node-local `/tmp/$USER/mds_cache`. `/tmp` and `/raid` are the same 28 TB md0 NVMe array and `/tmp` is already bound into the container.
  - Run `stage_mds.py` as a job prologue (`srun --ntasks-per-node=1`) with a process pool of about 32 × `zstd -d`.
  - Budget 2 TB per node. It completes in minutes at the rates measured in §1; confirm the aggregate `/scratch` read rate on the first real run.
  - Validate by comparing the file size with `raw_data.bytes` in `index.json`, so reruns on the same node skip already-staged shards.
- **Alternative:** decompress once to a shared `/scratch/.../mds_raw/` (+2 TB, about 10% more than today's zstd), if the project quota allows. Then no staging is needed and every node reads directly.
- Either way the old per-rank `predownload` / `cache_limit` machinery disappears.

### 4.6 Evaluation

- **During training:** loss only, on small fixed validation subsets per task/language. The old code used `epoch_size=512`; set `choose: 512` in a val YAML. Pass it as `--val_dataset`, which the same override handles.
  - Avoid `predict_with_generate` inside the trainer. It is slow for a 30B MoE.
- **Checkpoint eval:** a separate script that reads the same MDS val sets through the reader and generates with vLLM (the container's `swift infer --infer_backend vllm`, already tested). It scores with the old `metrics/wer.py`, `bleu.py` and `meteor.py` and the normalisers, weighted by the YAML `weight`. The numbers are then directly comparable with MERaLiON results.

## 5. Code sketches (not yet run)

```python
# omni_mds/reader.py
import json, os, struct, functools, numpy as np

class MDSShard:
    """Random access into one *uncompressed* MDS v2 shard."""
    def __init__(self, path, column_names, column_encodings):
        self.fd = os.open(path, os.O_RDONLY)
        n = struct.unpack('<I', os.pread(self.fd, 4, 0))[0]
        self.offsets = np.frombuffer(os.pread(self.fd, 4 * (n + 1), 4), np.uint32)
        self.cols, self.encs = column_names, column_encodings   # all columns here are variable-width (str/bytes)

    def __getitem__(self, i):
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        b = os.pread(self.fd, end - start, start)
        k = len(self.cols)
        sizes = np.frombuffer(b, np.uint32, k)
        out, pos = {}, 4 * k
        for name, enc, sz in zip(self.cols, self.encs, sizes):
            v = b[pos:pos + sz]; pos += sz
            out[name] = v.decode('utf-8') if enc == 'str' else v
        return out

@functools.lru_cache(maxsize=512)           # per dataloader worker
def open_shard(path, cols, encs):
    return MDSShard(path, list(cols), list(encs))
```

```python
# omni_mds/dataset.py
class MDSMixDataset(torch.utils.data.Dataset):
    def __init__(self, mix_yaml, index_dir, cache_dir, split='train', seed=42, epochs=1, row_fn=None): ...
        # loads parquet indexes, applies choose/repeat per virtual epoch → self.items: int64[L, 3]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        if i == 'lengths':                       # LazyLLMDataset forwards str keys; used by group_by_length
            return self.lengths
        d, s, r = self.items[i]
        raw = open_shard(*self.shard_key(d, s))[r]
        return self.row_fn(raw, meta=self.meta[d], rng=self._rng(i))
```

```python
# train_omni_mds.py  — used as: python train_omni_mds.py {sft|megatron} --dataset mixes/sea_v1.yaml --val_dataset mixes/sea_val.yaml ...
import sys
from swift.pipelines import SwiftSft
from omni_mds.dataset import MDSMixDataset

def make(base):
    class MDSSft(base):
        def _get_dataset(self):
            a = self.args
            assert a.lazy_tokenize and not a.streaming
            train = MDSMixDataset(a.dataset[0], split='train', seed=a.data_seed, epochs=int(a.num_train_epochs or 1))
            val = MDSMixDataset(a.val_dataset[0], split='val', seed=a.data_seed) if a.val_dataset else None
            return train, val
    return MDSSft

if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'megatron':
        from swift.megatron.pipelines.train.sft import MegatronSft as Base
    else:
        Base = SwiftSft
    make(Base)(sys.argv[1:]).main()
```

Check during the prototype:
- `_save_val_dataset`, `_show_dataset` and `_stat_dataset` expect an HF `Dataset` in places. With `lazy_tokenize=True` they skip the stat call; verify no other `isinstance(HfDataset)` path runs.
- For Megatron with `num_train_epochs`, check that `len(train)` drives `train_iters` as expected.
- Pass `--split_dataset_ratio 0` so ms-swift does not try to split our object.

## 6. Container

Use `container/swift_megatron_cu128.sif` for all multi-GPU work. It is the image the 8-GPU benchmark (job 147727) uses, and it has the same ms-swift commit as `toolkits/ms-swift`. Put the adapter code on `/scratch` (already bound) and import it through `PYTHONPATH`. That needs two small edits to `run_container.sh`:
1. Add `PYTHONPATH SWIFT_AUDIO_LOAD_BACKEND` to the `FORWARD=(…)` list. `--cleanenv` drops the host environment otherwise.
2. Nothing for the cache if it lives under `/tmp`, which is already bound. Use `EXTRA_BINDS=/raid` if you prefer `/raid`.

Keep the nemo_env/`swift_omni` venv for single-GPU debugging only. It has transformers 5.8.dev, while the container has 5.2.0. Do not compare numbers across the two stacks.

`mosaicml-streaming` is **not** needed for option D. If option C is ever needed, install it with `pip install --target /scratch/.../pydeps` and add that to `PYTHONPATH`, instead of rebuilding the image.

## 7. Plan

| # | step | output | check |
|---|---|---|---|
| 1 | `omni_mds/reader.py` + `build_mds_index.py`; index one WAV set and one Opus set | parquet indexes | sample-for-sample equality with mosaic-free decode (the peek script from §1); durations match `soundfile` |
| 2 | Port `row_fn` (normaliser + prompts + dropout), with unit tests on ~50 rows per task | `omni_mds/sea_text.py` | Diff targets against `InstructDataCollator._standardize_response` on the same rows |
| 3 | `MDSMixDataset` + `train_omni_mds.py`; 1-GPU `swift sft` LoRA smoke test on a 3-dataset mix | loss curve | `template.print_inputs` shows `<audio>` expanded to 13 tokens/s; no LoRA in `audio_tower` (explicit target list) |
| 4 | `stage_mds.py`; 8-GPU run in the container | samples/s, dataloader wait time | GPU util vs. the JSONL benchmark from job 147727; kill and resume to confirm the same next batch |
| 5 | Build indexes for the full accessible mix (~1 h); fix or remove the 78 missing SQA/AC paths | `mixes/sea_v1.yaml` | Per-task/language hour table, which feeds the data inventory (WORKLOG "Tomorrow" #4) |
| 6 | Megatron path (`train_omni_mds.py megatron`) once the trainer decision is made | | Same smoke test with EP=8 |
| 7 | vLLM eval script over the MDS val sets with the old metrics | per-task WER/MER/BLEU | Zero-shot baseline (guide §1) |

Steps 1–3 are about 1–2 days of work. After step 3 the choice of trainer (swift vs. Megatron) no longer affects the data code.

## 8. What this keeps and what it loses

The adapter replaces only the dataset object that ms-swift reads from. Everything after `template.encode` is the stock ms-swift / Megatron-SWIFT code that we already smoke-tested in the container. This holds by construction, but has not yet been verified with the adapter (plan step 3).

**Kept from ms-swift / Megatron-SWIFT** (all tested 2026-09-24 with JSONL data):
- the Qwen3-Omni template: audio tokens, chat format, labels;
- LoRA with the explicit target list, with no LoRA inside the audio encoder;
- flash-attn and gradient checkpointing;
- Megatron-SWIFT expert parallelism, and the PEFT adapter export;
- inference with `swift infer` and vLLM;
- resume and checkpointing.

The grouped_mm vs. default-experts vs. EP=8 comparison (job 147727) also carries over, because it does not depend on the data code.

**Changed or not yet covered:**
- Megatron `--packing true` needs every sample's length up front. That needs a small `PackingDataset` subclass fed from the index (§4.4). Until then, use padding-free.
- The 147727 numbers were measured with LibriSpeech JSONL. Re-check the dataloader wait time with the adapter (plan step 4).

**Old `multimodal_trainer` features that ms-swift does not replace:**

| old trainer feature | ms-swift equivalent | plan |
|---|---|---|
| Generation eval per validation set during training, with its own metric per set (WER/MER, BLEU, METEOR; `trainer.py:_evaluate_one_dataset`) | One `val_dataset`, `predict_with_generate` with `eval_metric` `nlg`/`acc` only; no WER; slow for a 30B MoE | Eval outside training: a vLLM script over the MDS val sets, run on each saved checkpoint (plan step 7) |
| Weighted aggregate across val sets (YAML `weight`), `best/` checkpoint chosen by it, best predictions saved | `metric_for_best_model` on one scalar | The same eval script writes the weighted score and keeps a `best` symlink |
| Saved predictions for every eval | none built in | The eval script writes JSONL predictions for every checkpoint |
| Mosaic mid-epoch resume | skip-batches (HF) / `consumed_samples` (Megatron) | Equivalent, given a deterministic index (§4.2) |
| FSDP2 for dense LLMs | DeepSpeed / FSDP via accelerate, Megatron | Not needed for the MoE; Megatron EP is the scaled path |
