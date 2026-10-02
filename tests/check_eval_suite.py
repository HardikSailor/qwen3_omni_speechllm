"""CPU check of an eval suite: every set loads, every item has audio + prompt + answer, and scoring the references as
predictions gives a perfect score (WER 0, BLEU 100) for the non-judge metrics. Inside the container:
    $RUN python tests/check_eval_suite.py [evals/suite_v0.yaml] [--n 4]"""
import argparse
import io
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf  # noqa: E402

from omni_eval import bench  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('suite', nargs='?', default='evals/suite_v0.yaml')
ap.add_argument('--n', type=int, default=4)
args = ap.parse_args()

import time  # noqa: E402

from omni_eval import compat  # noqa: E402

compat.install()
print('AudioBench-SEA optional normaliser libraries:', compat.sea_optional_libs(), flush=True)
bad = 0
for s in bench.load_suite(args.suite, 'quick'):
    t0 = time.time()
    s.n = args.n
    try:
        items = bench.build_items(s)
        assert len(items) == args.n, f'{len(items)} items'
        for it in items:
            assert it['text'] and it['answer'], 'empty prompt or answer'
            a = it['audio']
            info = sf.info(io.BytesIO(a)) if isinstance(a, bytes) else sf.info(a)
            assert info.duration > 0, 'empty audio'
        recs = [{**{k: v for k, v in it.items() if k != 'audio'}, 'model_prediction': it['answer']} for it in items]
        res = []
        for m in s.metrics:
            if m in bench.JUDGE_METRICS:
                res.append(f'{m}: (judge, skipped)')
                continue
            v = bench.headline(bench.score(s, recs, m), m)
            ok = v < 1e-6 if m in bench.RATE_METRICS else v > 99.0
            assert ok, f'{m} = {v} on the references themselves'
            res.append(f'{m}={v:.3g}')
        it = items[0]
        print(f'OK   {s.name:32s} {s.processor:10s} {info.duration:5.1f}s  {"; ".join(res)}  ({time.time() - t0:.0f} s)\n'
              f'       prompt: {it["text"][:110]!r}\n       answer: {it["answer"][:110]!r}', flush=True)
    except Exception as e:
        bad += 1
        print(f'FAIL {s.name}: {e!r}', flush=True)
        traceback.print_exc()
print(f'{bad} failures')
sys.exit(1 if bad else 0)
