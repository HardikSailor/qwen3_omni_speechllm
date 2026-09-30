# The training container

One container image holds the whole environment for fine-tuning and running Qwen3-Omni-30B-A3B: ms-swift (`swift sft`),
Megatron-SWIFT (`megatron sft`) and vLLM. No conda env or venv is needed. Last verified on NSCC: 2026-09-30.

## Where things are (NSCC)

Small files (scripts, recipe, docs) are in git, in this folder. Big files are **not** in git; they live in `CONTAINER_HOME`:

```
CONTAINER_HOME = /scratch/users/astar/ares/sailorhb/container          (the launchers read this variable)
├── swift_megatron_cu128.sqsh          enroot image, 12.6 GB       <- what NSCC runs (enroot is the only runtime on the compute nodes)
├── models--Qwen--Qwen3-Omni-30B-A3B-Instruct/   HF-cache layout, 66 GB; pass ".../snapshots/<hash>" to --model
├── pydeps/                            pure-Python extras that are NOT in the image (see below)
├── cache/nltk_data/                   NLTK wordnet, omw-1.4, punkt_tab (METEOR; compute nodes are offline)
├── cache/<user>/                      ModelScope / dataset cache, shared by all nodes
├── wheels_pydeps/                     the wheels pydeps was installed from
└── smoke_outputs/, logs/, bench_8gpu/ results of earlier checks
```

Data (MDS shards): `/data/projects/13003558/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal` (read only for us).

| In git (`container/`) | What |
|---|---|
| `run_container.sh` | **The launcher. Every job goes through it.** Picks Apptainer or enroot (NSCC has only enroot) |
| `run_container_enroot.sh` | enroot version: same interface and environment |
| `swift_megatron_cu128.def` | Apptainer recipe: what was added on top of the base image |
| `build_container.sh` | builds the `.sif` from the recipe (needs Apptainer; not available on NSCC) |
| `sif_to_enroot.sh` | `.sif` -> `.sqsh` (the NSCC image was produced this way, on the previous cluster) |
| `smoke_test.sh` | `env / infer / vllm / sft / megatron` checks. `infer`, `vllm`, `sft`, `megatron` need the LibriSpeech ASR smoke data, which is not on NSCC yet; `env` works |

## What is in the image

| Component | Version |
|---|---|
| Base image | `modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3` |
| Python / torch / CUDA | 3.11 / 2.10.0 / 12.8 |
| vLLM / flash-attn / DeepSpeed / apex | 0.17.1 / 2.8.3 / 0.18.8 / built from source |
| transformers / Transformer Engine | 5.2.0 / 2.13.0 |

**Added on top of the base image** (`swift_megatron_cu128.def`; base packages pinned so the CUDA stack stays as shipped):

| Package | Version | Why |
|---|---|---|
| ms-swift | 4.6.0.dev0 (git submodule `third_party/ms-swift` @ `8ec0455`, installed from `/opt/ms-swift`) | trainer, templates, `megatron sft` |
| megatron-core | 0.16.1 | Megatron-SWIFT backend. Not 0.17+: it needs Python >= 3.12 |
| mcore-bridge | 1.6.4 | loads HF weights into Megatron and exports adapters |
| datasets | 4.8.4 | |
| peft / trl | 0.18.1 / 0.28.0 | LoRA (ranges allowed by the recipe: peft >=0.11,<0.21; trl >=0.15,<1.0) |
| qwen-omni-utils, jiwer | 0.0.9, 4.0.0 | audio/video loading for Qwen-Omni; WER |

The full list is `transfer/requirements_container_freeze.txt` (406 packages), and the tested top-level pins are in `transfer/requirements.txt`.

**Outside the image, in `CONTAINER_HOME/pydeps`** (put on `PYTHONPATH` by `nscc/env.sh`):

| Package | Version | Why it is not in the image |
|---|---|---|
| mosaicml-streaming | 0.13.0 | reads the MDS shards. It declares `transformers<5` and cloud SDKs, so it is installed `--no-deps` |
| zstd, python-snappy, cramjam, catalogue | 1.5.7.2, 0.7.3, 2.12.1, 2.0.10 | its runtime dependencies (MDS codecs) |

The list is `transfer/requirements_pydeps.txt`. To recreate `pydeps` on NSCC: compute nodes cannot reach PyPI and the container has no proxy,
so download the wheels on the **login node** and install them offline inside the container:
```bash
pip download --no-deps --python-version 3.11 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform any \
    --implementation cp --abi cp311 --abi none -d $CONTAINER_HOME/wheels_pydeps -r transfer/requirements_pydeps.txt
# on a GPU node:
source nscc/env.sh
$RUN pip install --no-deps --no-index --target $CONTAINER_HOME/pydeps $CONTAINER_HOME/wheels_pydeps/*.whl
```

Known: `pip check` inside the image reports conflicts inherited from the base image (vLLM 0.17.1 declares `transformers<5`; lmdeploy / evalscope pins).
vLLM inference on Qwen3-Omni works regardless (tested 2026-09-24 on the previous cluster); lmdeploy and evalscope are untested.

## Using it

```bash
source nscc/env.sh                     # sets P C MODEL MIX RUN PYTHONPATH (see nscc/README.md)
$RUN python -c "import swift; print(swift.__version__)"
NPROC_PER_NODE=8 $RUN python train_omni.py megatron ...       # Megatron: always set NPROC_PER_NODE, even =1
```
Run this on a GPU node (an interactive PBS job or `qsub`), not on the login node.

What the launcher does:
- Starts the image with a clean environment (`enroot start`, or `apptainer exec --cleanenv`) and a clean home on node-local `/tmp/$USER/container_home`, so `~/.local` and `~/.bashrc` never leak in.
- Sets `PYTHONNOUSERSITE=1`, `HF_HOME`, `HF_HUB_OFFLINE=1`, `ENABLE_AUDIO_OUTPUT=0` (talker not loaded), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NLTK_DATA`, `MODELSCOPE_CACHE`.
- Forwards only these host variables: `CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT`, NCCL / GLOO ones (`NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_DISABLE NCCL_IB_GID_INDEX NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_P2P_LEVEL GLOO_SOCKET_IFNAME`), `PYTHONPATH`, `MODEL`, `MIX` (so a config can say `model: ${MODEL}`), `WANDB_*`, and our `OMNI_*` variables. Anything else you export on the host is **not** visible inside.
- Mounts `/scratch`, `/data/projects/13003558` and `/tmp` (host `/tmp`, because enroot's default `/tmp` is RAM). Add more with `EXTRA_BINDS=/a,/b:/c`.
- Overridable: `CONTAINER_HOME`, `SQSH` / `SIF`, `HF_MODELS`, `SHARED_CACHE`, `CONTAINER_RUNTIME=enroot|apptainer`.

enroot on NSCC:
- The first call in a job unpacks the image (~30 GB) into `ENROOT_DATA_PATH`. NSCC's default is a per-job folder on the node's `/raid` (`/raid/local/containers/enroot-data/$PBS_JOBID`); this takes about 15 s and needs no cleanup.
- Multi-node: every node unpacks its own copy (a lock protects against parallel starts on one node).
- Mellanox hook (`99-mellanox.sh`) is enabled on NSCC; whether InfiniBand is used from inside the container is checked by `nscc/nccl_check.py` in `pbs/train_32gpu_4node.pbs`.

## Checks

`bash container/smoke_test.sh env` (1 GPU node) prints versions and runs a Transformer Engine layer. Results on NSCC, H100 80 GB, driver 595.71:

| Test | Result (2026-09-29/30) |
|---|---|
| `env` | torch 2.10+cu128 sees the GPU; TE, megatron-core 0.16.1, ms-swift 4.6.0.dev0, vLLM 0.17.1 import |
| `train_omni.py sft` LoRA r16, 10 steps + resume | loss 1.06 -> ~1.1, 60.3 GiB peak, ~3 s/step; resume diff 0.007 |
| `train_omni.py megatron` LoRA r16, EP=1, 20 iterations + resume + 80 GB cap | 61.5 GiB peak, ~5 s/iter; resume diff 0.020 |
| `pbs/smoke_audio_lora_2gpu.pbs` (both trainers, 2 GPUs, audio-encoder LoRA) | pass; HF 61.7 GiB, Megatron EP=2 34.4 GiB |
| `tests/test_mosaic_stream.py` | 7/8 pass; the 2-rank test needs >= 2 GPUs on the host (NCCL "Duplicate GPU" on a 1-GPU job) and passed in the 2-GPU job |

Earlier results on the previous cluster (H200, Apptainer, 2026-09-24): see `PIPELINE_PLAN.md` §10 and `DECISION_LOG.md`.

## Rebuilding and upgrading

- **The NSCC image was not built on NSCC**: it is the `.sqsh` converted from the Apptainer `.sif` built on the previous cluster. NSCC compute nodes have enroot and no Apptainer.
- To change the image: edit `swift_megatron_cu128.def`, build with `build_container.sh` on a machine with Apptainer (~15 min, ~50 GB local disk; it snapshots `third_party/ms-swift`), convert with `sif_to_enroot.sh` (~1 min), copy the `.sqsh` to `CONTAINER_HOME`. Building the same recipe with enroot/docker from the base image URI is untested.
- **ms-swift upgrade rule** (D23): branch, move the submodule pin, rebuild, run the smoke tests (`pbs/smoke_audio_lora_2gpu.pbs` and the resume tests), then commit the pin together with the new image. Never commit a pin the image doesn't match.
- The image keeps the Megatron-SWIFT adapter quirk: `adapter_config.json` from a plain Megatron LoRA has bare `target_modules` (`q_proj`, ...). `train_omni.py` rewrites them to exact module names (see `PIPELINE_PLAN.md` §10.7).
