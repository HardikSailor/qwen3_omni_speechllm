#!/bin/bash
# Run a command inside the Qwen3-Omni training container (ms-swift + Megatron-SWIFT + vLLM, CUDA 12.8).
#
#   bash run_container.sh <command> [args...]
#   bash run_container.sh swift sft --model ... ...
#   bash run_container.sh megatron sft --model ... ...
#   bash run_container.sh python my_script.py
#
# Must run on a GPU node (inside an salloc/sbatch allocation). Multi-node: one `srun` task per node, e.g.
#   srun -N2 --ntasks-per-node=1 bash run_container.sh megatron sft ...
# NNODES / NODE_RANK / MASTER_ADDR are derived from Slurm below when not set.
#
# Overridable env vars:
#   CONTAINER_HOME   directory with the big files: .sif/.sqsh, model weights, pydeps, caches
#                    (default: /scratch/users/astar/ares/sailorhb/container; see container/README.md)
#   SIF              container image          (default: $CONTAINER_HOME/swift_megatron_cu128.sif)
#   HF_MODELS        HF cache with the weights (default: $CONTAINER_HOME; the model is passed as a local path)
#   SHARED_CACHE     dataset/modelscope cache on shared storage, needed for multi-node (default: $CONTAINER_HOME/cache/$USER)
#   EXTRA_BINDS      extra -B binds, comma separated (e.g. /data/mine:/data/mine)
#   CONTAINER_RUNTIME apptainer | enroot (default: apptainer if installed, else enroot via run_container_enroot.sh)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# No Apptainer here (e.g. an enroot-only cluster): hand over to the enroot launcher, same interface.
# Force either runtime with CONTAINER_RUNTIME=apptainer|enroot.
if [ "${CONTAINER_RUNTIME:-}" = enroot ] || { [ -z "${CONTAINER_RUNTIME:-}" ] && ! command -v apptainer >/dev/null 2>&1 \
        && command -v enroot >/dev/null 2>&1; }; then
    exec bash "$HERE/run_container_enroot.sh" "$@"
fi
export CONTAINER_HOME=${CONTAINER_HOME:-/scratch/users/astar/ares/sailorhb/container}
SIF=${SIF:-$CONTAINER_HOME/swift_megatron_cu128.sif}
HF_MODELS=${HF_MODELS:-$CONTAINER_HOME}
SHARED_CACHE=${SHARED_CACHE:-$CONTAINER_HOME/cache/$USER}
# Clean, writable per-user home on node-local disk: keeps ~/.local packages and ~/.bashrc out of the container.
CHOME=${CHOME:-/tmp/$USER/container_home}
mkdir -p "$SHARED_CACHE" "$CHOME"

# Multi-node rendezvous from Slurm (ms-swift reads torchrun-style variables).
if [ -n "$SLURM_JOB_NODELIST" ] && [ -z "$MASTER_ADDR" ]; then
    export NNODES=${NNODES:-$SLURM_JOB_NUM_NODES}
    export NODE_RANK=${NODE_RANK:-$SLURM_NODEID}
    SCONTROL=$(command -v scontrol || echo /cm/shared/apps/slurm/current/bin/scontrol)
    export MASTER_ADDR=$($SCONTROL show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
    export MASTER_PORT=${MASTER_PORT:-29500}
fi
[ "${NNODES:-1}" = 1 ] && unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT

# --cleanenv drops the host environment; forward only what training needs.
FORWARD=(GLOO_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_P2P_LEVEL TORCH_DISTRIBUTED_DEBUG CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
         NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_DISABLE WANDB_API_KEY WANDB_MODE
         MAX_PIXELS VIDEO_MAX_PIXELS FPS_MAX_FRAMES IMAGE_MAX_TOKEN_NUM VIDEO_MAX_TOKEN_NUM
         TORCH_NCCL_ASYNC_ERROR_HANDLING CUDA_DEVICE_MAX_CONNECTIONS
         PYTHONPATH SWIFT_AUDIO_LOAD_BACKEND OMNI_GPU_MEM_GB OMNI_MDS_CACHE OMNI_DEBUG_OPTIM_LOAD OMNI_LOG_SAMPLE_IDS)
ENVFILE=$(mktemp "$CHOME/.envfile.XXXXXX")
{
    echo "PYTHONNOUSERSITE=1"
    echo "HF_HOME=$HF_MODELS"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    echo "MODELSCOPE_CACHE=$SHARED_CACHE"
    echo "ENABLE_AUDIO_OUTPUT=${ENABLE_AUDIO_OUTPUT:-0}"
    echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    echo "TRITON_CACHE_DIR=$CHOME/.triton"
    echo "TMPDIR=${TMPDIR:-/tmp}"
    # METEOR (omni_mds/metrics) needs NLTK wordnet/omw; pre-downloaded here because compute nodes may be offline.
    echo "NLTK_DATA=${NLTK_DATA:-$CONTAINER_HOME/cache/nltk_data}"
    for v in "${FORWARD[@]}"; do [ -n "${!v}" ] && echo "$v=${!v}"; done
} > "$ENVFILE"

BINDS="/scratch,/data/projects/13003558,/tmp"
[ -n "$EXTRA_BINDS" ] && BINDS="$BINDS,$EXTRA_BINDS"

apptainer exec --nv --cleanenv --home "$CHOME" --env-file "$ENVFILE" -B "$BINDS" --pwd "$PWD" "$SIF" "$@"
rc=$?
rm -f "$ENVFILE"
exit $rc
