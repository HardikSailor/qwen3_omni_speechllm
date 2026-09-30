#!/bin/bash
# Per-node wrapper for multi-node PBS jobs. Run on EVERY node (via pbsdsh) with the same arguments:
#   node_run.sh <NODES_FILE> <LOG_PREFIX> <MASTER_PORT> <NPROC_PER_NODE> -- <command...>
# NODES_FILE = one unique hostname per line (the first is the rendezvous host). Sets NNODES, NODE_RANK, MASTER_ADDR/PORT,
# then runs the command (normally `$RUN python train_omni.py ...`). Rank 0 logs to <LOG_PREFIX>.log, the others to
# <LOG_PREFIX>.node<K>.log, so tests/summarize_smoke.py finds the usual <phase>.log. DRY_RUN=1 prints the environment and command.
NODES_FILE=$1; LOG=$2; PORT=$3; NPROC=$4; shift 5
mapfile -t NODES < <(awk '{sub(/\..*/,"")} !s[$0]++' "$NODES_FILE")
ME=${NODE_RUN_HOST:-$(hostname -s)}; RANK=-1   # NODE_RUN_HOST: only for dry runs
for i in "${!NODES[@]}"; do [ "${NODES[$i]}" = "$ME" ] && RANK=$i; done
[ "$RANK" -ge 0 ] || { echo "node_run: $ME not in $NODES_FILE (${NODES[*]})" >&2; exit 97; }
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export NNODES=${#NODES[@]} NODE_RANK=$RANK MASTER_ADDR=${NODES[0]} MASTER_PORT=$PORT NPROC_PER_NODE=$NPROC
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC-1)))
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
[ "$RANK" = 0 ] && OUTF=$LOG.log || OUTF=$LOG.node$RANK.log
cd "$P"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[dry] $ME rank=$RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT nproc=$NPROC gpus=$CUDA_VISIBLE_DEVICES log=$OUTF"; echo "[dry] $*"; exit 0
fi
"$@" > "$OUTF" 2>&1; rc=$?
echo "[node_run] $ME rank=$RANK exit=$rc"
exit $rc
