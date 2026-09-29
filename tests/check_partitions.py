"""Check per-rank sample-id logs written with OMNI_LOG_SAMPLE_IDS=<dir> (mosaic_stream.py).

    python tests/check_partitions.py <dir>
Every rank must read a disjoint set of samples, and the ranks must get (almost) equal numbers. Worker prefetching
reads a little ahead of what training consumed, so counts can differ by a few batches per worker.
"""
import collections, glob, os, sys

d = sys.argv[1]
per_rank = collections.defaultdict(list)
for f in glob.glob(os.path.join(d, 'rank*_pid*.txt')):
    rank = int(os.path.basename(f).split('_')[0][4:])
    per_rank[rank] += [int(x) for x in open(f) if x.strip()]
if not per_rank:
    sys.exit(f'no sample-id files in {d}')
sets = {r: set(v) for r, v in per_rank.items()}
dups_within = {r: len(v) - len(sets[r]) for r, v in per_rank.items()}
overlap = sum(len(sets[a] & sets[b]) for a in sets for b in sets if a < b)
counts = {r: len(v) for r, v in sorted(per_rank.items())}
print(f'ranks: {len(per_rank)}  samples per rank: {counts}')
print(f'repeated ids within a rank: {sum(dups_within.values())}   ids shared between ranks: {overlap}')
print('OK: partitions are disjoint' if overlap == 0 else 'FAIL: ranks share samples')
