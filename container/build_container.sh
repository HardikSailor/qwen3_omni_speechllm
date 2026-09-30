#!/bin/bash
# Build swift_megatron_cu128.sif on a compute node. All temporary files go to node-local /tmp;
# only the final image (~13 GB) is copied to this directory on /scratch.
#   bash container/build_container.sh   (needs Apptainer; NSCC compute nodes only have enroot, so build where Apptainer exists, then sif_to_enroot.sh)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_HOME=${CONTAINER_HOME:-/scratch/users/astar/ares/sailorhb/container}   # the .sif is written here
W=/tmp/sailorhb_apptainer
BASE_URI=docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3
export APPTAINER_CACHEDIR=$W/cache APPTAINER_TMPDIR=$W/tmp
mkdir -p $W/cache $W/tmp $W/out $W/src

[ -f $W/out/swift_base_cu128.sif ] || apptainer pull $W/out/swift_base_cu128.sif $BASE_URI

# Snapshot of the local ms-swift source (no .git, no build artifacts).
rsync -a --delete --exclude .git --exclude '__pycache__' --exclude '*.egg-info' --exclude build \
    "$HERE/../third_party/ms-swift/" $W/src/ms-swift/    # the git submodule, pinned to the commit in the image

apptainer build --force $W/out/swift_megatron_cu128.sif $HERE/swift_megatron_cu128.def
cp $W/out/swift_megatron_cu128.sif $CONTAINER_HOME/swift_megatron_cu128.sif.part
mv $CONTAINER_HOME/swift_megatron_cu128.sif.part $CONTAINER_HOME/swift_megatron_cu128.sif
ls -la $CONTAINER_HOME/swift_megatron_cu128.sif
