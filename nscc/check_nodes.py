#!/usr/bin/env python3
"""How many reservation nodes are free right now for a PBS script? Runs on the login node (pbs_rstat + pbsnodes only).
  python3 nscc/check_nodes.py [pbs/train_mv4_lora_v0.pbs] [--exclude a2ap-dgx011,...]
Reads the queue (-q R...) and the chunk count (select=<N>) from the script, lists the reservation's nodes, and counts
those with no jobs on them (whole-node chunks need an empty node), minus the excluded (broken) ones. Prints a ready
`-l select=1:host=...+...` for qsub when enough are free. Other users' jobs are invisible to qstat but pbsnodes shows them.
"""
import argparse, json, re, subprocess, sys

BAD = ["a2ap-dgx011"]   # no /dev/infiniband/rdma_cm (enroot mellanox hook fails), 2026-10-02

ap = argparse.ArgumentParser()
ap.add_argument("pbs", nargs="?", default="pbs/train_mv4_lora_v0.pbs")
ap.add_argument("--exclude", default=",".join(BAD), help="comma-separated broken nodes")
a = ap.parse_args()
bad = {h for h in a.exclude.split(",") if h}

text = open(a.pbs).read()
queue = re.search(r"^#PBS\s+-q\s+(\S+)", text, re.M).group(1)
sel = re.search(r"^#PBS\s+-l\s+select=(\d+):(\S+)", text, re.M)
need, chunk = int(sel.group(1)), sel.group(2)

rstat = subprocess.run(["pbs_rstat", "-f", queue], capture_output=True, text=True).stdout
resv = re.findall(r"\(([\w-]+):", re.search(r"resv_nodes = (.*)", rstat).group(1))
state = re.search(r"reserve_state = (\S+)", rstat).group(1)

nodes = json.loads(subprocess.run(["pbsnodes", "-F", "json", *resv], capture_output=True, text=True).stdout)["nodes"]
free = []
print(f"{a.pbs}: queue {queue} ({state}), needs {need} x [{chunk}]\n")
print(f"{'node':14} {'state':18} jobs")
for h in resv:
    n = nodes.get(h, {})
    st = n.get("state", "unknown")
    jobs = sorted({j.split("/")[0] for j in n.get("jobs", [])}) if n.get("jobs") else []
    ok = st == "free" and not jobs and h not in bad
    note = "  <- excluded (bad)" if h in bad else ""
    print(f"{h:14} {st:18} {','.join(jobs) or '-'}{note}{'  FREE' if ok else ''}")
    if ok: free.append(h)

print(f"\nfree now: {len(free)} / {len(resv)} reservation nodes (excluding {','.join(sorted(bad)) or 'none'}); "
      f"script needs {need}")
if free:
    k = min(len(free), need)
    while 256 % (8 * k): k -= 1   # global batch 256 = 8 GPUs x K nodes x ACCUM (x per-device 1)
    acc = f"  -v ACCUM={256 // (8 * k)}" if k != need else ""
    hosts = "+".join(f"1:host={h}:{chunk}" for h in free[:k])
    print(f"{'enough' if len(free) >= need else 'NOT enough; with what is free'}:\n"
          f"  qsub -l select={hosts}{acc} {a.pbs}")
sys.exit(0 if len(free) >= need else 1)
