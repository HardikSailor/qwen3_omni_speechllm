#!/bin/bash
# Run ONCE on the new cluster after copy_to_h100.sh. It rewrites the hard-coded Orion paths in scripts and configs.
#
#   bash relocate_paths.sh <new_root> <new_datasets_multimodal>
#   e.g. bash relocate_paths.sh /home/project/12345/omni /home/project/12345/datasets_multimodal
#
# <new_root> is the directory that holds container/, toolkits/ and hf_models/ (= DEST in copy_to_h100.sh).
# It rewrites .py/.sh/.sbatch/.yaml files under toolkits/qwen3_omni_speechllm and container/, and keeps a *.orig copy of each
# changed file. Markdown docs are left alone. It also points the container's bind mounts at the new paths instead of /scratch.
set -eu
NEW_ROOT=$(realpath "${1:?usage: relocate_paths.sh <new_root> <new_datasets_multimodal>}")
NEW_DATA=$(realpath "${2:?usage: relocate_paths.sh <new_root> <new_datasets_multimodal>}")
OLD_ROOT=/scratch/prj0000000234/sailorhb
OLD_DATA=/scratch/prj0000000234/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal
[ -d "$NEW_ROOT/toolkits/qwen3_omni_speechllm" ] || { echo "no toolkits/qwen3_omni_speechllm under $NEW_ROOT"; exit 1; }

FILES=$(grep -rlE "$OLD_ROOT|$OLD_DATA|/scratch,/tmp" "$NEW_ROOT/toolkits/qwen3_omni_speechllm" "$NEW_ROOT/container" \
        --include='*.py' --include='*.sh' --include='*.sbatch' --include='*.yaml' \
        --exclude-dir=transfer --exclude-dir=env_backup --exclude-dir=third_party || true)
for f in $FILES; do
    [ -f "$f.orig" ] || cp -p "$f" "$f.orig"
    sed -i -e "s#$OLD_DATA#$NEW_DATA#g" -e "s#$OLD_ROOT#$NEW_ROOT#g" "$f"
    echo "rewrote $f"
done
# Both launchers (Apptainer and enroot) bind /scratch and /tmp; /scratch may not exist on the new cluster.
for f in run_container.sh run_container_enroot.sh; do
    [ -f "$NEW_ROOT/container/$f" ] || continue
    sed -i "s#^BINDS=\"/scratch,/tmp\"#BINDS=\"$NEW_ROOT,$NEW_DATA,/tmp\"#" "$NEW_ROOT/container/$f"
    grep -Hn '^BINDS=' "$NEW_ROOT/container/$f"
done

echo; echo "Remaining references to /scratch (should only be docs or legacy scripts):"
grep -rn "/scratch/" "$NEW_ROOT/toolkits/qwen3_omni_speechllm" "$NEW_ROOT/container" \
     --include='*.py' --include='*.sh' --include='*.sbatch' --include='*.yaml' --exclude-dir=transfer --exclude-dir=env_backup --exclude-dir=third_party || echo "  none"
