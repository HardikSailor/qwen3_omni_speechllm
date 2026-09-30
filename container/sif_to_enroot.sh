#!/bin/bash
# Convert swift_megatron_cu128.sif (Apptainer) into swift_megatron_cu128.sqsh (enroot), for clusters without Apptainer.
# Nothing is rebuilt or downloaded: the SIF's root filesystem is reused as is. Only enroot's own files are added:
#   /etc/environment  <- the image's Docker Env (PATH, CUDA_VERSION, NVIDIA_* ...) + PYTHONNOUSERSITE=1 (the .def's %environment)
#   /etc/rc           <- runs the given command (like the /etc/rc that `enroot import` writes)
# Needs: apptainer (for `sif dump`), unsquashfs, mksquashfs, python3, ~40 GB of local disk (use node-local /tmp).
#   bash sif_to_enroot.sh [in.sif] [out.sqsh]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_HOME=${CONTAINER_HOME:-/scratch/users/astar/ares/sailorhb/container}
SIF=${1:-$CONTAINER_HOME/swift_megatron_cu128.sif}
OUT=${2:-${SIF%.sif}.sqsh}
W=${WORK:-/tmp/$USER/sif_to_enroot}
rm -rf "$W"; mkdir -p "$W"

# The SIF partition of type FS (Squashfs) is the root filesystem; the JSON partition with "Env" is the image config.
FS_ID=$(apptainer sif list "$SIF" | awk -F'|' '/FS \(Squashfs/ {gsub(/ /,"",$1); print $1}')
echo "root filesystem = SIF partition $FS_ID"
apptainer sif dump "$FS_ID" "$SIF" > "$W/rootfs.sqsh"
ENV_JSON=
for id in $(apptainer sif list "$SIF" | awk -F'|' '/JSON.Generic/ {gsub(/ /,"",$1); print $1}'); do
    apptainer sif dump "$id" "$SIF" > "$W/part$id.json"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if 'Env' in d.get('config', d) else 1)" \
        "$W/part$id.json" && ENV_JSON="$W/part$id.json"
done
[ -n "$ENV_JSON" ] || { echo "no image config with Env found in $SIF"; exit 1; }

unsquashfs -q -no-xattrs -d "$W/rootfs" "$W/rootfs.sqsh" || echo "(unsquashfs warnings above are expected for device files as non-root)"
rm -f "$W/rootfs.sqsh"
python3 - "$ENV_JSON" "$W/rootfs/etc/environment" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
env = d.get('config', d)['Env']   # SIF stores the OCI config at top level; Docker configs nest it
env = [e for e in env if not e.startswith('PYTHONNOUSERSITE=')] + ['PYTHONNOUSERSITE=1']
open(sys.argv[2], 'w').write('\n'.join(env) + '\n')
print(f'/etc/environment: {len(env)} variables')
EOF
cat > "$W/rootfs/etc/rc" <<'EOF'
#!/bin/sh
# enroot entry: run the given command, or a shell.
[ $# -gt 0 ] && exec "$@"
exec /bin/bash
EOF
chmod 755 "$W/rootfs/etc/rc"

mksquashfs "$W/rootfs" "$W/out.sqsh" -noappend -no-xattrs -all-root -comp zstd -Xcompression-level 3 -processors "$(nproc)" -quiet
cp "$W/out.sqsh" "$OUT.part" && mv "$OUT.part" "$OUT"
rm -rf "$W"
ls -la "$OUT"

