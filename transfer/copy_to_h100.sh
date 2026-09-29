#!/bin/bash
# Copy everything the Qwen3-Omni toolkit needs to another cluster (e.g. the H100 machines). See TRANSFER_TO_H100.md.
#
#   DEST=user@h100-login:/path/to/root  DATA_DEST=user@h100-login:/path/to/datasets_multimodal \
#       bash transfer/copy_to_h100.sh [code] [container] [model] [data]        # dry run: prints what would be copied
#   IMAGE=sqsh RUN=1 DEST=... bash transfer/copy_to_h100.sh container          # enroot image instead of the .sif
#   RUN=1 DEST=...  DATA_DEST=...  bash transfer/copy_to_h100.sh ...            # really copy
#
# With no part names, it copies all four. rsync is resumable: re-run the same command after an interruption.
# DEST gets the same layout as /scratch/prj0000000234/sailorhb (container/, toolkits/, hf_models/), so after
# copying you only need to run transfer/relocate_paths.sh once on the new cluster.
set -eu
SRC=/scratch/prj0000000234/sailorhb
SRC_DATA=/scratch/prj0000000234/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal
: "${DEST:?set DEST=[user@host:]/new/root}"
PARTS=("$@"); [ ${#PARTS[@]} -eq 0 ] && PARTS=(code container model data)
OPTS=(-aH --partial --mkpath)
if [ "${RUN:-0}" = 1 ]; then OPTS+=(--info=progress2); else OPTS+=(--dry-run --itemize-changes); echo "DRY RUN (set RUN=1 to copy)"; fi

sync() { echo ">>> $1  ->  $2"; rsync "${OPTS[@]}" "${@:3}" "$1" "$2"; }

for p in "${PARTS[@]}"; do case $p in
  code)       # ~3 MB code + 14 MB pydeps + 120 MB ms-swift source (provenance / rebuilds)
    sync $SRC/toolkits/qwen3_omni_speechllm/ "$DEST/toolkits/qwen3_omni_speechllm/" \
         --exclude outputs/ --exclude logs/ --exclude __pycache__/ --exclude '*.pyc'
    sync $SRC/toolkits/pydeps/   "$DEST/toolkits/pydeps/"
    sync $SRC/toolkits/ms-swift/ "$DEST/toolkits/ms-swift/" --exclude __pycache__/ --exclude '*.egg-info' --exclude build/ ;;
  container)  # image (IMAGE=sif|sqsh|both, default sif; sqsh = enroot, made by container/sif_to_enroot.sh) + NLTK data + launchers
    case ${IMAGE:-sif} in sif) IMG=(--include swift_megatron_cu128.sif) ;; sqsh) IMG=(--include swift_megatron_cu128.sqsh) ;;
         both) IMG=(--include swift_megatron_cu128.sif --include swift_megatron_cu128.sqsh) ;; *) echo "IMAGE=sif|sqsh|both"; exit 1 ;; esac
    sync $SRC/container/ "$DEST/container/" \
         --include run_container.sh --include run_container_enroot.sh --include sif_to_enroot.sh \
         --include build_container.sh --include smoke_test.sh --include README.md \
         --include swift_megatron_cu128.def "${IMG[@]}" \
         --include cache/ --include cache/nltk_data/ --include 'cache/nltk_data/**' --exclude '*' ;;
  model)      # 66 GB HF cache for Qwen3-Omni-30B-A3B-Instruct (keeps the blobs/ + snapshots/ symlink layout)
    sync $SRC/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/ \
         "$DEST/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/" ;;
  data)       # MDS datasets named in mixes/st_v0.yaml, ~195 GB (gigaspeech en-zh 107 GB, people's speech en-ms 73 GB)
    : "${DATA_DEST:?set DATA_DEST=[user@host:]/new/datasets_multimodal}"
    for d in $(grep -E '^\s+path:' $SRC/toolkits/qwen3_omni_speechllm/mixes/st_v0.yaml | awk '{print $2}'); do
        sync $SRC_DATA/$d/ "$DATA_DEST/$d/"
    done ;;
  *) echo "unknown part '$p' (use: code container model data)"; exit 1 ;;
esac; done
echo "done. On the new cluster: bash <root>/toolkits/qwen3_omni_speechllm/transfer/relocate_paths.sh <root> <datasets_multimodal>"
