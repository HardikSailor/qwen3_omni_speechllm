# Moving the Qwen3-Omni toolkit to the H100 cluster

Written 2026-09-29. The whole environment is **one Apptainer image**. You do not need a conda env or venv. Copy the image, the code, the model and the data, then rewrite paths once.

## 1. What to copy (~275 GB total)

| Part | Source on Orion (`/scratch/prj0000000234/...`) | Size | Needed? |
|---|---|---|---|
| **code** | `sailorhb/toolkits/qwen3_omni_speechllm/` (without `outputs/`, `logs/`, `__pycache__/`) | 3.5 MB | yes |
| | `sailorhb/toolkits/pydeps/`: MDS reader (mosaicml-streaming 0.13 + codecs), not in the image | 14 MB | yes |
| | `sailorhb/toolkits/ms-swift/`: source baked into the image (commit `8ec045582`) | 120 MB | only to rebuild the image |
| **container** | `sailorhb/container/swift_megatron_cu128.sif`: torch 2.10 / CUDA 12.8, ms-swift, Megatron-SWIFT, vLLM | 12.6 GB | yes |
| | `sailorhb/container/run_container.sh`: launcher that every job script calls | – | yes |
| | `sailorhb/container/cache/nltk_data/`: wordnet, omw-1.4, punkt_tab for METEOR (compute nodes are offline) | 51 MB | yes |
| | `sailorhb/container/{swift_megatron_cu128.def, build_container.sh, README.md, smoke_test.sh}` | – | rebuild / checks |
| **model** | `sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/` (HF cache layout, symlinks) | 66 GB | yes |
| **data** | `zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal/{train,test}/ST/...`: the 14 MDS folders named in `mixes/st_v0.yaml` | ~195 GB | yes (or point to an existing copy) |

**Don't copy:**
- `toolkits/venvs/swift_omni`: a Python 3.10 venv layered on the `nemo_env` conda env in `$HOME`; it can't be moved. It is only used by the old 1-GPU scripts `train_lora_asr_smoke.sh` and `run_env_check.sh`.
- `outputs/` (3.5 GB of smoke-test checkpoints).
- `toolkits/Megatron-LM` and `toolkits/multimodal_trainer`: the image has megatron-core, and the needed multimodal_trainer files are vendored in `omni_mds/`.

Data: the old MERaLiON trainer on ASPIRE2A+ read MDS data from `/data/projects/13003558`. If the same `datasets_multimodal` tree is already there, skip `data` and point `relocate_paths.sh` at it.

## 2. Copy

From an Orion login node (rsync over ssh, resumable, dry run by default):

```bash
cd /scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
export DEST=user@h100-login:/path/to/omni_root              # gets container/ toolkits/ hf_models/
export DATA_DEST=user@h100-login:/path/to/datasets_multimodal
bash transfer/copy_to_h100.sh code container                # dry run: lists files
RUN=1 bash transfer/copy_to_h100.sh code container          # copy
RUN=1 bash transfer/copy_to_h100.sh model                   # 66 GB
RUN=1 bash transfer/copy_to_h100.sh data                    # ~195 GB; skip if the data is already there
```

If the H100 cluster blocks inbound ssh from Orion, run the copy from the other side (swap source and destination in the script's `rsync` calls) or use its data-transfer node.

## 3. Fix paths (once, on the H100 cluster)

```bash
bash /path/to/omni_root/toolkits/qwen3_omni_speechllm/transfer/relocate_paths.sh /path/to/omni_root /path/to/datasets_multimodal
```

The script:
- rewrites `/scratch/prj0000000234/sailorhb` and the dataset root in all `.py/.sh/.sbatch/.yaml` files (keeping `*.orig` backups);
- changes the container bind mounts from `/scratch,/tmp` to the new roots;
- lists any remaining `/scratch/` references.

It was tested on a local copy: no references remained.

## 4. Check the environment

```bash
C=/path/to/omni_root/container
apptainer exec $C/swift_megatron_cu128.sif python -c "import swift, megatron.core; print(swift.__version__)"   # login node, no GPU
bash $C/smoke_test.sh env      # on a GPU node: torch sees the GPUs, Transformer Engine runs
bash $C/smoke_test.sh infer    # 3 LibriSpeech clips (needs data/LibriSpeech, or edit the paths)
```

- **Apptainer/Singularity missing?** Build from the same base image with enroot/docker (the base image URI is in `requirements.txt`), then install the "step 2" packages. `swift_megatron_cu128.def` is the exact recipe.
- **Driver:** the image uses CUDA 12.8. Our H200 nodes run driver 565.57 (CUDA 12.7) fine through CUDA minor-version compatibility. Any driver that supports CUDA 12.x should work; check with `nvidia-smi`.

## 5. What changes on H100 (80 GB) — PIPELINE_PLAN.md §9

H100 and H200 are the same architecture (sm_90), so no code changes. Only memory differs:

| Setting | H200 (141 GB) | H100 (80 GB) |
|---|---|---|
| LoRA, HF DDP (`train_omni.py sft`) | 61.8 GiB peak | Fits with ~17 GiB spare. Keep micro-batch 1, gradient checkpointing, `--max_length 2048` |
| LoRA, Megatron EP=8 | 14.2 GiB | Same. **Preferred on H100** |
| vLLM eval | TP=1 | Use `tensor_parallel_size=2` |
| Full fine-tuning | 1 node borderline | ≥ 2 nodes |

- Keep global batch 32: with 8 GPUs use grad-accum 4; with 4 GPUs use grad-accum 8 (`slurm/train_4gpu_both.sbatch`).
- To test the 80 GB budget on our H200s first, set `OMNI_GPU_MEM_GB=80`.
- Node-local `/tmp` holds the MDS cache (`OMNI_MDS_CACHE`) and the container home. If `/tmp` is small on the H100 nodes, set `OMNI_MDS_CACHE` and `CHOME` to a larger local disk.
- The job scripts in `slurm/` use `#SBATCH`. If the H100 cluster runs PBS (ASPIRE2A+), convert the header (`#PBS -l select=1:ngpus=8`, ...). The body works as is.

## 6. State of the code at copy time (2026-09-29)

- Both trainers pass on 1, 2 and 4 GPUs: training, eval, resume, and disjoint data partitions (`PIPELINE_PLAN.md` §10.3–10.7).
  - Useful job scripts: `slurm/train_4gpu_both.sbatch` (4 GPUs) and `slurm/train_8gpu_both.sbatch` (8 GPUs, never run yet).
- New options:
  - `--audio_lora_layers` / `--audio_lora_modules`: LoRA inside the audio encoder (`PIPELINE_PLAN.md` §10.7);
  - smoke test: `slurm/smoke_audio_lora_2gpu.sbatch`.
- After `relocate_paths.sh`, the first check on the new cluster: `slurm/smoke_audio_lora_2gpu.sbatch` (≈ 5 min on 2 GPUs).
  - It runs both trainers.
  - Convert its header to PBS if needed.
  - Its `--val_choose 8` keeps eval short.
- Background: `DECISION_LOG.md` (D19–D21 for this move), `WORKLOG.md`.

## 7. Requirement files in this folder

| File | Use |
|---|---|
| `requirements.txt` | Pinned top-level packages, grouped in install order. Only needed if you must rebuild without the `.sif` |
| `requirements_pydeps.txt` | The MDS reader extras in `toolkits/pydeps`. Install with `--no-deps --target` inside the container |
| `requirements_container_freeze.txt` | Full `pip freeze` of the image (406 packages), for reference and diffs |
