#!/bin/bash
# enroot version of run_container.sh, for clusters without Apptainer. Same usage, same environment inside:
#   bash run_container_enroot.sh <command> [args...]
# run_container.sh calls this automatically when `apptainer` is missing and `enroot` exists (or CONTAINER_RUNTIME=enroot).
#
# Overridable env vars (in addition to those of run_container.sh):
#   CONTAINER_HOME    directory with the .sqsh, model, pydeps, caches (default: /scratch/users/astar/ares/sailorhb/container)
#   SQSH              enroot image (default: $CONTAINER_HOME/swift_megatron_cu128.sqsh; make it with sif_to_enroot.sh)
#   ENROOT_DATA_PATH  where the image is unpacked once per node (~30 GB). Default: the system setting if it is not in
#                     $HOME, else /tmp/$USER/enroot/data (node-local)
#   EXTRA_BINDS       extra host paths, comma separated (src or src:dst); /scratch and /tmp are always mounted
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTAINER_HOME=${CONTAINER_HOME:-/scratch/users/astar/ares/sailorhb/container}
SQSH=${SQSH:-$CONTAINER_HOME/swift_megatron_cu128.sqsh}
HF_MODELS=${HF_MODELS:-$CONTAINER_HOME}
SHARED_CACHE=${SHARED_CACHE:-$CONTAINER_HOME/cache/$USER}
CHOME=${CHOME:-/tmp/$USER/container_home}
mkdir -p "$SHARED_CACHE" "$CHOME"

# Unpack the image once per node, on local disk (never into $HOME), keyed by the image's size + mtime.
if [ -z "${ENROOT_DATA_PATH:-}" ]; then
    sys=$(eval echo "$(awk '$1=="ENROOT_DATA_PATH" {$1=""; print}' /etc/enroot/enroot.conf 2>/dev/null)")
    case "$sys" in ""|"$HOME"*) export ENROOT_DATA_PATH=/tmp/$USER/enroot/data ;; *) export ENROOT_DATA_PATH=$sys ;; esac
fi
mkdir -p "$ENROOT_DATA_PATH"
NAME=omni_$(stat -c '%s_%Y' "$SQSH")
(
    flock 9
    if ! enroot list | grep -qx "$NAME"; then
        echo "[run_container_enroot] unpacking $SQSH as $NAME in $ENROOT_DATA_PATH (once per node)" >&2
        enroot create --name "$NAME" "$SQSH" >&2 || exit 1
    fi
) 9>"$ENROOT_DATA_PATH/.omni_create.lock" || exit 1

# Multi-node rendezvous from Slurm (as in run_container.sh).
if [ -n "$SLURM_JOB_NODELIST" ] && [ -z "$MASTER_ADDR" ]; then
    export NNODES=${NNODES:-$SLURM_JOB_NUM_NODES}
    export NODE_RANK=${NODE_RANK:-$SLURM_NODEID}
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
    export MASTER_PORT=${MASTER_PORT:-29500}
fi
[ "${NNODES:-1}" = 1 ] && unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT

# Explicit environment, as with apptainer --cleanenv (same list as run_container.sh).
FORWARD=(GLOO_SOCKET_IFNAME NCCL_IB_GID_INDEX NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_P2P_LEVEL TORCH_DISTRIBUTED_DEBUG CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
         NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_DISABLE WANDB_API_KEY WANDB_MODE
         MAX_PIXELS VIDEO_MAX_PIXELS FPS_MAX_FRAMES IMAGE_MAX_TOKEN_NUM VIDEO_MAX_TOKEN_NUM
         TORCH_NCCL_ASYNC_ERROR_HANDLING CUDA_DEVICE_MAX_CONNECTIONS
         PYTHONPATH MODEL MIX SWIFT_AUDIO_LOAD_BACKEND OMNI_GPU_MEM_GB OMNI_MDS_CACHE OMNI_DEBUG_OPTIM_LOAD OMNI_LOG_SAMPLE_IDS)
ENVS=(-e HOME="$CHOME" -e PYTHONNOUSERSITE=1 -e HF_HOME="$HF_MODELS" -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
      -e MODELSCOPE_CACHE="$SHARED_CACHE" -e ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT:-0}"
      -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      -e TRITON_CACHE_DIR="$CHOME/.triton" -e TMPDIR="${TMPDIR:-/tmp}"
      -e NLTK_DATA="${NLTK_DATA:-$CONTAINER_HOME/cache/nltk_data}")
for v in "${FORWARD[@]}"; do [ -n "${!v}" ] && ENVS+=(-e "$v=${!v}"); done

# enroot mounts a RAM tmpfs on /tmp by default; binding the host /tmp keeps the mosaic cache on local disk.
BINDS="/scratch,/data/projects/13003558,/tmp"
[ -n "$EXTRA_BINDS" ] && BINDS="$BINDS,$EXTRA_BINDS"
MOUNTS=()
IFS=',' read -ra B <<< "$BINDS"
for b in "${B[@]}"; do
    [ -z "$b" ] && continue
    case "$b" in *:*) MOUNTS+=(-m "$b") ;; *) MOUNTS+=(-m "$b:$b") ;; esac
done

exec enroot start "${ENVS[@]}" "${MOUNTS[@]}" "$NAME" \
    /bin/bash -c 'cd "$1" && shift && exec "$@"' _ "$PWD" "$@"
