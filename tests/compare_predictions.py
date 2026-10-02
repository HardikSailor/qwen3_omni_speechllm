"""Compare two eval_omni.py prediction dirs set by set (same idx -> same prediction?). Used to check that a checkpoint
switched in after others (load-once adapter switching) gives exactly what a fresh load gives.
    python tests/compare_predictions.py <dir A> <dir B>     # exit 1 if any prediction differs"""
import glob
import json
import os
import sys

a_dir, b_dir = sys.argv[1:3]
same = diff = 0
for f in sorted(glob.glob(os.path.join(b_dir, '*.json'))):
    name = os.path.basename(f)
    if name in ('meta.json', 'scores.json') or '_score' in name or not os.path.exists(os.path.join(a_dir, name)):
        continue
    a = {r['idx']: r['model_prediction'] for r in json.load(open(os.path.join(a_dir, name)))}
    for r in json.load(open(f)):
        if a.get(r['idx']) == r['model_prediction']:
            same += 1
        else:
            diff += 1
            if diff <= 5:
                print(f'DIFF {name} {r["idx"]}: {a.get(r["idx"], "")[:80]!r} | {r["model_prediction"][:80]!r}')
print(f'{same} identical, {diff} different predictions ({a_dir} vs {b_dir})')
sys.exit(1 if diff or not same else 0)
