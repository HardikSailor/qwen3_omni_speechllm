"""Summarise smoke runs (slurm/smoke_1gpu.sbatch, slurm/smoke_megatron_1gpu.sbatch).

    python tests/summarize_smoke.py <out_dir> [RUN_A RUN_B RUN_C]      (default: A B C)
RUN_A = full run, RUN_B = resumed from RUN_A's middle checkpoint (losses must match), RUN_C = 80 GB-cap run.
Works for HF (`swift sft`) and Megatron logging.jsonl files.
"""
import glob
import json
import os
import sys

out = sys.argv[1]
runs = sys.argv[2:5] if len(sys.argv) >= 5 else ['A', 'B', 'C']


def logs(run):
    rows = []
    for f in sorted(glob.glob(f'{out}/{run}/**/logging.jsonl', recursive=True)):
        rows += [json.loads(line) for line in open(f) if line.strip()]
    return rows


def step_of(r):
    if 'global_step/max_steps' in r:
        return int(str(r['global_step/max_steps']).split('/')[0])
    for k in ('iteration', 'global_step', 'step'):
        if k in r:
            return int(str(r[k]).split('/')[0])
    return None


def steps(rows):
    res = {}
    for r in rows:
        s = step_of(r)
        if s is not None and 'loss' in r and not any(k.startswith('eval') for k in r):
            res[s] = r
    return res


def num(r, *keys):
    for k in keys:
        if k in r:
            try:
                return float(str(r[k]).split()[0])
            except ValueError:
                pass
    return float('nan')


for run in runs:
    rows = logs(run)
    st = steps(rows)
    if not st:
        print(f'{run}: no training log')
        continue
    ks = sorted(st)
    mem = max(num(r, 'memory(GiB)', 'max_memory_allocated(GiB)', 'memory') for r in st.values())
    speed = [num(st[k], 'train_speed(s/it)', 'elapsed_time_per_iteration(s)', 'elapsed_time_per_iteration')
             for k in ks if k > 3]
    ev = [r for r in rows if any(k.startswith('eval') and 'loss' in k for k in r)]
    ev_losses = [round(num(e, 'eval_loss', 'eval/loss', 'eval_lm loss', 'eval_loss_lm'), 4) for e in ev]
    print(f'{run}: steps {ks[0]}..{ks[-1]}  loss first {st[ks[0]]["loss"]:.4f} last {st[ks[-1]]["loss"]:.4f}  '
          f'peak mem {mem:.1f} GiB  s/it (last) {speed[-1] if speed else float("nan"):.2f}  eval losses {ev_losses}')

a, b = steps(logs(runs[0])), steps(logs(runs[1]))
common = sorted(set(a) & set(b))
if common:
    diffs = [abs(a[k]['loss'] - b[k]['loss']) for k in common]
    print(f'resume check ({runs[0]} vs {runs[1]}, steps {common[0]}..{common[-1]}): '
          f'max |loss diff| = {max(diffs):.5f}')
    for k in common[:5]:
        print(f'  step {k}: {runs[0]} {a[k]["loss"]:.4f}  {runs[1]} {b[k]["loss"]:.4f}')
else:
    print(f'resume check: no common steps between {runs[0]} and {runs[1]}')

files = sorted(glob.glob(f'{out}/{runs[0]}/**/adapter_model.safetensors', recursive=True))
if files:
    from safetensors import safe_open
    f = files[-1]
    sf = safe_open(f, 'pt')
    keys = list(sf.keys())
    enc = [k for k in keys if 'audio_tower.layers' in k]
    print(f'adapter {os.path.relpath(f, out)}: {len(keys)} tensors, '
          f'{len([k for k in keys if "audio_tower" in k])} in audio_tower, {len(enc)} in audio_tower.layers '
          f'(0 unless --audio_lora_layers is set)')
    if enc:
        layers = sorted({int(k.split('audio_tower.layers.')[1].split('.')[0]) for k in enc})
        mods = sorted({k.split('audio_tower.layers.')[1].split('.', 1)[1].rsplit('.lora_', 1)[0] for k in enc})
        b = [k for k in enc if 'lora_B' in k]
        trained = sum(bool(sf.get_tensor(k).abs().max() > 0) for k in b)
        print(f'  audio LoRA layers {layers}\n  modules {mods}\n  lora_B non-zero (trained): {trained}/{len(b)}')
    cfg = os.path.join(os.path.dirname(f), 'adapter_config.json')
    if os.path.isfile(cfg):
        tm = json.load(open(cfg)).get('target_modules')
        print(f'  adapter_config target_modules: {len(tm) if isinstance(tm, list) else tm!r} entries, '
              f'e.g. {sorted(tm)[:3] if isinstance(tm, list) else ""}')
for f in sorted(glob.glob(f'{out}/{runs[0]}/**/omni_mosaic_state.json', recursive=True)):
    print(f'{os.path.relpath(f, out)}: {open(f).read().strip()}')
