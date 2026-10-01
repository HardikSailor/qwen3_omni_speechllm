# Qwen3-Omni speech LLM for Southeast Asia and Singapore

Fine-tuning **Qwen3-Omni-30B-A3B-Instruct** (a 30.5B MoE thinker with a 32-layer audio encoder) for multi-task audio understanding: ASR, speech translation, emotion, spoken QA and audio events, including code-switching and SEA languages.

The pipeline is **stock [ms-swift](https://github.com/modelscope/ms-swift) plus our own data layer**. Training reads the existing MERaLiON **MDS shards** directly, with no conversion step. The per-sample prompt and target logic is ported exactly from the old `multimodal_trainer`. Both of ms-swift's trainers are supported:

| Trainer | Command | Parallelism | Best for |
|---|---|---|---|
| HF (`swift sft`) | `train_omni.py sft` | DDP | LoRA prototyping; plain PyTorch, easy to modify |
| Megatron-SWIFT | `train_omni.py megatron` | Expert parallel (EP) | Large runs, expert LoRA, full fine-tuning, 80 GB GPUs |

**Status (2026-09-30), running on NSCC (H100 80 GB, PBS, enroot):**
- Both trainers pass on 1 and 2 H100 GPUs here (train, eval, resume; audio-encoder LoRA); on 1, 2, 4 and 8 H200 GPUs on the previous cluster.
- Scripts for 1 node x 8 GPUs and 4 nodes x 8 GPUs are ready for the dedicated queue; not yet run.
- LoRA inside the audio encoder is supported. The eval script is next.

The code moved from the previous cluster (Orion, Slurm, H200) to NSCC; older docs describe Orion, with paths mapped in [Paths](#paths). See `WORKLOG.md` for the daily log.

---

## Contents

- [How it works](#how-it-works)
- [Paths](#paths)
- [Quick start](#quick-start)
- [Training options](#training-options)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Results so far](#results-so-far)
- [Deployment](#deployment)
- [Documentation map](#documentation-map)
- [Gotchas](#gotchas)

---

## How it works

```
 mixes/*.yaml ──► omni_mds.mix ──► mosaic streaming reader ──► sea_text.build_row ──► ms-swift template.encode
 (datasets,       (dataset specs)   (per-GPU partitions,       (prompt + target,        (audio → 13 tokens/s,
  choose, lang)                      resumable position)        ported 1:1)              padding-free batches)
                                                                                                 │
                         ┌───────────────────────────────────────────────────────────────────────┘
                         ▼
     train_omni.py sft       → ms-swift HF trainer (DDP)       ┐  checkpoints = PEFT adapter
     train_omni.py megatron  → Megatron-SWIFT (EP, mcore)      ┘  + omni_mosaic_state.json (data position)
```

`train_omni.py` subclasses ms-swift and doesn't patch it. It replaces only:
- dataset preparation;
- the dataloaders (each GPU reads its own mosaic partition);
- checkpoint saving (the data position is stored, so resume continues exactly where it stopped).

Everything else is ordinary ms-swift arguments: model loading, LoRA, template, collator, optimizer, saving.

## Paths

| What | NSCC location |
|---|---|
| This repo | `/scratch/users/astar/ares/sailorhb/git_repos/qwen3_omni_speechllm` |
| Container image, model, `pydeps`, caches (`CONTAINER_HOME`) | `/scratch/users/astar/ares/sailorhb/container/` (see `container/README.md`) |
| Model weights | `$CONTAINER_HOME/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/<hash>` |
| MDS data | `/data/projects/13003558/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal` |
| Outputs, logs | `outputs/` in the repo (git-ignored), `outputs/pbs_logs/` for PBS job logs |

Older docs use the previous cluster's paths: `/scratch/prj0000000234/sailorhb/{toolkits,container,hf_models}` and `/scratch/prj0000000234/zoux/datasets/...`.
Read `toolkits/qwen3_omni_speechllm` as this repo, `container` and `hf_models` as `CONTAINER_HOME`, and `zoux/datasets/...` as the `/data/projects/13003558/zoux/datasets/...` above.

## What you need to run it

- A GPU node: an interactive PBS job (`qsub -I -q ... -l select=1:ngpus=1:ncpus=14:mem=235GB -P 13003558`) or `qsub pbs/<job>.pbs`. Never run on the login node.
- enroot (the only container runtime on NSCC compute nodes) and the image in `CONTAINER_HOME`. What the image contains and what was added on top: `container/README.md`.
- The `pydeps` folder with `mosaicml-streaming` (the MDS reader, not in the image). Recreate: `container/README.md`.
- Model weights and MDS data at the paths above; read access needs the project group `13003558`.
- The submodule: `git clone --recursive` (reference only; the code that runs is the copy inside the image).

## Quick start

**Get the code.** ms-swift is a git submodule pinned to the exact commit in the container:
```bash
git clone --recursive git@github.com:HardikSailor/qwen3_omni_speechllm.git
# already cloned without --recursive:
git submodule update --init
```

Everything runs inside one container image; you don't need a conda env or venv. `nscc/env.sh` sets the paths (`P C MODEL MIX RUN PYTHONPATH`),
and `$RUN` is `container/run_container.sh`. Run the commands below on a GPU node.

```bash
cd /scratch/users/astar/ares/sailorhb/git_repos/qwen3_omni_speechllm
source nscc/env.sh
export NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3        # the script re-launches under torchrun
```

**HF trainer, LoRA on thinker attention + audio projector, global batch 32:**
```bash
$RUN python train_omni.py sft \
    --mix $MIX --val_choose 32 \
    --model "$MODEL" --model_type qwen3_omni_moe --experts_impl grouped_mm \
    --tuner_type lora --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) --lora_rank 16 --lora_alpha 32 \
    --torch_dtype bfloat16 --attn_impl flash_attn --padding_free true \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --gradient_checkpointing true --max_length 2048 \
    --max_steps 1000 --save_steps 200 --eval_steps 200 --output_dir outputs/my_run
```

**Megatron-SWIFT, expert parallel over 4 GPUs:**
```bash
$RUN python train_omni.py megatron \
    --mix $MIX --val_choose 32 \
    --model "$MODEL" --model_type qwen3_omni_moe \
    --tuner_type lora --target_modules linear_qkv linear_proj --lora_rank 16 --lora_alpha 32 \
    --expert_model_parallel_size 4 --moe_grouped_gemm true --moe_permute_fusion true \
    --micro_batch_size 1 --global_batch_size 32 --lr 1e-4 \
    --train_iters 1000 --save_steps 200 --eval_steps 200 --finetune true --output_dir outputs/my_meg_run
```

**Job scripts:** `qsub pbs/<name>.pbs` (add `-q dedicated` for the dedicated queue). `train_4gpu_both.pbs`, `train_8gpu_both.pbs` (one node) and `train_32gpu_4node.pbs` (4 nodes x 8 GPUs) run both trainers with resume checks. Details: `nscc/README.md`. The scripts take their training settings from `configs/` and list only their own differences.

**Resume:**
- HF: `--resume_from_checkpoint <ckpt>`.
- Megatron: `--mcore_adapter <ckpt> --finetune false`, and save with `--no_save_optim false` if you want the optimizer state back (D17).

## Training options

**Recipe files.** Settings can live in a YAML file: `configs/lora_ddp.yaml` (HF trainer) and `configs/lora_megatron.yaml`
(Megatron) hold the full LoRA recipe, with learning rate, weight decay, LoRA rank / alpha / dropout, batch, schedule and the
ms-swift defaults written out. Copy one per experiment and pass it with `--config`; a flag on the command line replaces
the value in the file:
```bash
$RUN python train_omni.py sft --config configs/lora_ddp.yaml --output_dir outputs/my_run
NPROC_PER_NODE=8 $RUN python train_omni.py megatron --config configs/lora_megatron.yaml --finetune true --output_dir outputs/my_meg_run
```
- Keys are flag names without `--`: any `swift sft` / `megatron sft` argument plus the options below. The two trainers name
  some settings differently (`learning_rate` vs `lr`, `max_steps` vs `train_iters`, `max_grad_norm` vs `clip_grad`, ...).
- Lists become several values, `true`/`false` booleans, `null` drops a key, `${VAR}` comes from the environment
  (`model: ${MODEL}`, set by `nscc/env.sh`), `target_modules_file:` reads the LoRA targets from a file.
- **What was actually used:** every run writes `args.json` (ms-swift's final arguments) and `omni_effective.json` (read back
  from the live optimizer, scheduler and LoRA layers). `python tests/check_effective_args.py <config> <run>/v0-...` compares all
  three per key. The audit `nscc/audit_config_1gpu.sh` (2026-09-30) ran both trainers with non-default values: all matched.

`train_omni.py` accepts every `swift sft` / `megatron sft` argument, plus these:

| Option | Default | Meaning |
|---|---|---|
| `--mix` | *(required)* | Mix YAML: datasets, `choose` / `repeat`, languages, eval weights (`mixes/st_v0.yaml`) |
| `--val_choose N` | – | Override the validation size per dataset (short evals) |
| `--audio_lora_layers` | `none` | LoRA inside the audio encoder: `all`, `top<K>`, `bottom<K>`, `a-b`, `i,j,k` (32 layers, 0-based) |
| `--audio_lora_modules` | `attn` | `attn` (q/k/v/out_proj), `mlp` (fc1/fc2), `attn_mlp` |
| `--asr_prompt_type`, `--ac_prompt_type`, `--english_prompt_weight`, `--augment_prompt`, `--instruction_dropout_rate`, `--asr_normalize_text`, `--augment_audio` | as in the old trainer | Per-sample prompt and augmentation settings (`omni_mds/sea_text.py`) |
| `--omni_cache_root`, `--cache_limit`, `--shuffle_block_size`, `--num_canonical_nodes` | – | Mosaic reader settings |
| `--wandb` | `$WANDB`, else `false` | Log to Weights & Biases (both trainers), see below |
| `--wandb_project`, `--wandb_entity`, `--wandb_run_name`, `--wandb_group`, `--wandb_tags` | `$WANDB_PROJECT` or `meralion_v4`; `$WANDB_ENTITY` or `i2r-llm`; basename of `--output_dir`; none; none | Where the run goes and how it is named (each also read from the matching `WANDB_*` variable) |
| `--wandb_mode` | online | `offline` writes to `<output_dir>/wandb`; upload later with `wandb sync <dir>` |
| `--wandb_resume` | `auto` | On resume: `auto` continues the checkpoint's run only when the checkpoint is inside `--output_dir` (a restart), else starts a new run in the same group; `always`; `never` |

Environment variables:
- `OMNI_GPU_MEM_GB=80` caps GPU memory (emulates an H100 on an H200; harmless on H100).
- `OMNI_LOG_SAMPLE_IDS=<dir>` logs which samples each GPU reads (used by `tests/check_partitions.py`).

Guards added automatically unless you pass the flag yourself:
- `--save_total_limit 3`;
- `--merge_lora false` for Megatron (its default writes a 66 GB merged copy per checkpoint).

**Weights & Biases.** `--wandb true` (or `WANDB=1`, as in the old shell scripts) adds `wandb` to `--report_to` and lets
each trainer log through its own callback: `swift sft` on rank 0 every `--logging_steps` (x axis `train/global_step`),
Megatron on the last rank every `--log_interval` (x axis = iteration). One run per launch, named after `--output_dir`; the run
config holds the trainer arguments plus our options, the mix and `omni_effective.json`. The run id is saved as
`omni_wandb.json` in every checkpoint, so a restart from a checkpoint inside the same `--output_dir` continues the same run
(for Megatron, W&B drops the points between the checkpoint and the last logged iteration). Checkpoints are not uploaded
(`WANDB_LOG_MODEL=false`). The API key comes from `WANDB_API_KEY`, else `~/.netrc` (read by `nscc/env.sh`); compute nodes
reach `api.wandb.ai` through the site proxy, which the container launcher forwards. PBS training scripts: `qsub -v WANDB=1 ...`
puts every phase of the job in one group `<job name>_<job id>`.

**Audio-encoder LoRA example:** add LoRA to the attention and MLP of the last 8 encoder layers:
```bash
... --audio_lora_layers top8 --audio_lora_modules attn_mlp --target_modules <explicit list> ...
```
It needs an explicit `--target_modules` list; `all-linear` and `--target_regex` are refused. Details: `PIPELINE_PLAN.md` §10.7.

## Repository layout

```
train_omni.py                 entry point: `sft` (HF) or `megatron`, plus the data/resume hooks
omni_mds/                     data layer
  mix.py                        mix YAML → dataset specs
  mosaic_stream.py              mosaic streaming reader: per-GPU partitions, resumable, padding-free counting
  sea_text.py                   build_row: prompts and targets per task (exact port of the old collators)
  mds_io.py, audio/             MDS decoding, audio loading and augmentation
  instructions_lib/, text_normalizers/, metrics/   copied from multimodal_trainer (see VENDORED.md)
configs/                      training recipes (YAML) for --config: lora_ddp.yaml, lora_megatron.yaml
mixes/st_v0.yaml              ST-only pilot mix (8 train + 6 test CoVoST2 / GigaSpeech / People's Speech sets)
lora_targets_thinker_attn_audio_proj.txt   explicit LoRA targets: 48 thinker attention layers + audio proj1/proj2
pbs/                          PBS job scripts for NSCC: smoke tests, 4/8-GPU runs, 4-node x 8-GPU run, Megatron resume debugging
nscc/                         NSCC settings (env.sh), per-node launcher for multi-node jobs, NCCL check; see nscc/README.md
container/                    launchers, image recipe, container docs (the image itself lives in CONTAINER_HOME); see container/README.md
slurm/                        LEGACY: the same jobs as Slurm scripts for the previous cluster (Orion); pbs/ is the current set
tests/                        CPU tests and run summarisers
tools/vendored/               old MDS conversion and sampling scripts (reference)
transfer/                     LEGACY copy/relocate kit for the Orion -> H100 move (Orion paths); requirements*.txt are still current
docs/                         upstream bug report draft (Megatron optimizer resume)
third_party/ms-swift/         git submodule: ms-swift @ 8ec0455 (the version inside the container; reference + rebuilds)
env_backup/                   package lists of the earlier conda/venv setups (reference)
*.md                          plan, decisions, deployment, research guide, work log
infer_*.py, make_swift_asr_data.py, train_lora_asr_smoke.sh, run_env_check.sh   early (09-24) inference and LoRA checks
```

Outside this directory (big files, not in git): `CONTAINER_HOME=/scratch/users/astar/ares/sailorhb/container/` with the enroot image, model weights, `pydeps/` (mosaicml-streaming 0.13 and codecs, installed `--no-deps`) and caches. Contents and how to recreate them: `container/README.md`.

## Tests

| Test | Where | What it checks |
|---|---|---|
| `tests/test_sea_text.py` | CPU | Prompt/target parity with the old trainer (8,392/8,392), dropout, augmentation |
| `tests/test_mosaic_stream.py` | CPU | Reader: same samples, `choose` per epoch, fixed validation set, disjoint ranks (needs >= 2 GPUs on the host), exact resume |
| `tests/test_config.py`, `tests/test_wandb.py` | CPU | `--config` expansion and the pbs/ scripts' differences; `--wandb` arguments, environment and resume rules |
| `tests/check_encode_cpu.py` | CPU | Full data path without the model: 13 audio tokens/s, label masking, padding-free batches |
| `pbs/smoke_1gpu.pbs`, `smoke_megatron_1gpu.pbs` | 1 GPU | Train, eval, resume, 80 GB cap |
| `pbs/smoke_audio_lora_2gpu.pbs` | 2 GPUs, ~12 min | Both trainers with audio-encoder LoRA. **Good first check on a new cluster** |
| `pbs/train_4gpu_both.pbs`, `train_8gpu_both.pbs` | 4 / 8 GPUs | Both trainers, resume, partition check |
| `pbs/train_32gpu_4node.pbs` | 4 nodes x 8 GPUs | NCCL bandwidth, then both trainers on 32 GPUs with resume |
| `pbs/train_16gpu_2node.pbs` | 2 nodes x 8 GPUs | The same on 16 GPUs, reserved queue R212478; `qsub -v MIX_FILE=...,WANDB=1` |

`tests/summarize_smoke.py <out_dir> A B C` summarises a run. It prints loss, memory, speed, the resume loss difference, adapter contents (including which encoder layers have LoRA and whether they trained) and the saved data position.

## Results so far

| Run | HF DDP | Megatron |
|---|---|---|
| 4× H200, global batch 32, 60 steps (job 150431) | loss 1.175 → 0.831, 61.8 GiB, 4.8 s/step | EP=4: loss 1.172 → 0.837, **23.5 GiB**, 5.6 s/step |
| Resume from step 30 (max loss difference) | 0.004 | 0.003 (with optimizer) |
| Audio-encoder LoRA top4 × attn_mlp (job 150442) | 24/24 encoder `lora_B` trained | 24/24 trained |
| 8× H200 benchmark (job 147727, LibriSpeech) | 17.0 samples/s | EP=8: 13.5 samples/s, 14.2 GiB |

The two trainers report eval loss on different scales (1.31 vs. 0.80); compare eval losses within one trainer only.

## Deployment

**vLLM 0.17.1 cannot serve LoRA adapters for Qwen3-Omni** (`supports_lora` = False). What follows from that:
- For deployment, merge the adapter into the weights (`swift export --merge_lora`) and serve the merged model with stock vLLM.
- A partner running it on-prem needs only a GPU host, an NVIDIA driver, Docker, the pinned `vllm/vllm-openai` image and the model folder (~66 GB in bf16: 1× H200 or 2× 80 GB GPUs).
- Keep the architecture identical to the public model, so stock vLLM keeps working.

Details, sizing, the hand-off package and licences: **`DEPLOYMENT.md`**.

## Cluster history

- **Orion (Slurm, H200, Apptainer)**: where the pipeline was built and tested through 2026-09-29. Its scripts are kept as `slurm/` and `transfer/`, marked legacy.
- **NSCC (PBS, H100 80 GB, enroot)**: the current home, from 2026-09-29. `pbs/`, `nscc/` and `container/` are the current scripts.
  H100 memory guidance is in `PIPELINE_PLAN.md` §9. Megatron EP is preferred on 80 GB GPUs.

## Documentation map

| File | Read it for |
|---|---|
| `PIPELINE_PLAN.md` | Design, code provenance, parallelism, H100 portability, **progress and results (§10)** |
| `DECISION_LOG.md` | Why things are the way they are (D1–D25), bugs found (B1–B21), evidence per job |
| `DEPLOYMENT.md` | Serving, eval backends, partner on-prem hand-off, training rules for deployability |
| `FINETUNING_GUIDE.md` | Research plan: stages, SEA data, architecture ideas, RL, evaluation |
| `MDS_DATA_PIPELINE.md` | How the MDS data reaches ms-swift; streaming vs. map-style |
| `TRAINING_AND_LOCAL_VOCAB_EXPERIMENTS.md` | Grounded training and contextual-biasing experiments |
| `WORKLOG.md` | Day-by-day log |
| `omni_mds/VENDORED.md` | Files copied from the old trainer, with md5s |
| `container/README.md` | The container: where it is, versions, what was added, launcher, rebuilding |
| `nscc/README.md` | Running on NSCC: PBS scripts, multi-node design, dedicated-queue tests |

## Gotchas

- **Always launch through `run_container.sh`.** It isolates the container from `~/.local`, sets `HF_HOME`, offline mode and `NLTK_DATA`, and forwards the variables training needs. It works with Apptainer or enroot (`CONTAINER_RUNTIME` forces one); NSCC has only enroot. Host variables not in its forward list are invisible inside.
- **Megatron:**
  - `NPROC_PER_NODE` must be set, even to 1.
  - TP, PP and CP must stay 1, because the mosaic reader partitions by global rank. EP is fine.
  - `--save_total_limit` must be ≥ 2.
- **LoRA targets:** use explicit module lists. Regex targets like `q_proj|k_proj` also match the audio encoder (B1). After training, check the adapter with `summarize_smoke.py`.
- **Disk:** checkpoints are GBs; keep `outputs/` small. Never merge LoRA per checkpoint. Save optimizer state only when a run must be resumable (`PIPELINE_PLAN.md` §10.5).
- **Pinned versions:**
  - ms-swift is the submodule `third_party/ms-swift`, pinned at `8ec0455`, and `train_omni.py` depends on its internals (D23).
  - The code that actually runs is the copy installed in the container. The submodule records which version that is.
- **Upgrading ms-swift:**
  1. On a branch, move the pin: `git -C third_party/ms-swift fetch && git -C third_party/ms-swift checkout <commit>`.
  2. Rebuild the container from that source.
  3. Re-run `pbs/smoke_audio_lora_2gpu.pbs` and the resume tests.
  4. If they pass, commit the new pin together with the new image. Never commit a pin the container doesn't match.
