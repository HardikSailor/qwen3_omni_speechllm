"""Average the LoRA adapters of several checkpoints of one run (checkpoint averaging / "model soup" of a trajectory).

    $RUN python tools/average_lora.py --out <dir> [--mode concat|mean] <ckpt dir> <ckpt dir> ...
e.g. the last four mv4_lora_v0 checkpoints (pbs/eval_omni.pbs AVERAGE= does this before evaluating):
    $RUN python tools/average_lora.py --out $RUN_DIR/avg4-16000-18500-concat $RUN_DIR/checkpoint-{16000,17000,18000,18500}

A LoRA layer adds scale * B @ A to the frozen weight (scale = lora_alpha / r). Two ways to average k adapters:
  concat (default, exact): the average of the k weight updates, mean_i(B_i @ A_i), written as ONE LoRA of rank k*r:
      A' = [A_1; ...; A_k] (rows stacked), B' = [B_1, ..., B_k] / k (columns), r' = k*r, lora_alpha' = k*lora_alpha
      (same scale). B' @ A' = mean_i(B_i @ A_i) exactly, up to the stored dtype. Costs k times the adapter size.
  mean (approximate): A' = mean_i A_i, B' = mean_i B_i at rank r. (mean B)(mean A) != mean(B A); close only when the
      checkpoints are close (late in one run at a low LR), and wrong for adapters from different runs or seeds
      (their rank directions are not aligned).
The output is a normal PEFT adapter dir (adapter_config.json + adapter_model.safetensors + omni_average.json), so
eval_omni.py, swift infer and `--adapters` can load it. A self-check compares B' @ A' with mean_i(B_i @ A_i) on a few
modules and fails above a small tolerance.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load(ckpt):
    with open(os.path.join(ckpt, 'adapter_config.json')) as f:
        cfg = json.load(f)
    with safe_open(os.path.join(ckpt, 'adapter_model.safetensors'), 'pt') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata()
    return cfg, tensors, meta


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('ckpts', nargs='+', help='checkpoint dirs with adapter_config.json + adapter_model.safetensors')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['concat', 'mean'], default='concat')
    p.add_argument('--check_modules', type=int, default=8, help='modules compared in the self-check')
    args = p.parse_args()
    k = len(args.ckpts)
    if k < 2:
        sys.exit('average_lora.py: give at least two checkpoints')

    cfgs, weights, meta = [], [], None
    for c in args.ckpts:
        cfg, t, meta = load(c)
        cfgs.append(cfg)
        weights.append(t)
        print(f'[average_lora] {c}: {len(t)} tensors, r={cfg["r"]} alpha={cfg["lora_alpha"]}', flush=True)
    cfg = cfgs[0]
    for c, other, w in zip(args.ckpts[1:], cfgs[1:], weights[1:]):
        for key in ('r', 'lora_alpha', 'use_rslora', 'use_dora', 'target_modules', 'rank_pattern', 'alpha_pattern'):
            if other.get(key) != cfg.get(key):
                sys.exit(f'average_lora.py: {c}: {key} differs from {args.ckpts[0]}')
        if set(w) != set(weights[0]) or any(w[n].shape != weights[0][n].shape for n in w):
            sys.exit(f'average_lora.py: {c}: tensor names or shapes differ from {args.ckpts[0]}')
    if cfg.get('use_dora') or cfg.get('rank_pattern') or cfg.get('alpha_pattern') or cfg.get('modules_to_save'):
        sys.exit('average_lora.py: DoRA, rank/alpha patterns and modules_to_save are not handled')
    other = [n for n in weights[0] if '.lora_A.' not in n and '.lora_B.' not in n]
    if other:
        sys.exit(f'average_lora.py: non-LoRA tensors not handled: {other[:5]}')

    out = {}
    for name in weights[0]:
        ts = [w[name].float() for w in weights]
        if args.mode == 'mean':
            v = sum(ts) / k
        elif '.lora_A.' in name:   # (r, in) -> (k r, in)
            v = torch.cat(ts, dim=0)
        else:                      # lora_B (out, r) -> (out, k r), scaled by 1/k
            v = torch.cat(ts, dim=1) / k
        out[name] = v.to(weights[0][name].dtype).contiguous()

    new_cfg = dict(cfg)
    if args.mode == 'concat':
        new_cfg['r'] = cfg['r'] * k
        new_cfg['lora_alpha'] = cfg['lora_alpha'] * k

    # self-check: B' A' * scale' against mean_i(B_i A_i) * scale, relative error on a few modules
    scale, new_scale = cfg['lora_alpha'] / cfg['r'], new_cfg['lora_alpha'] / new_cfg['r']
    a_names = sorted(n for n in out if '.lora_A.' in n)
    step = max(1, len(a_names) // args.check_modules)
    worst = 0.0
    for an in a_names[::step][:args.check_modules]:
        bn = an.replace('.lora_A.', '.lora_B.')
        ref = sum(w[bn].float() @ w[an].float() for w in weights) / k * scale
        got = out[bn].float() @ out[an].float() * new_scale
        rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        worst = max(worst, rel)
        print(f'[average_lora] check {an.rsplit(".lora_A.", 1)[0]}: relative error {rel:.2e}', flush=True)
    tol = 2e-2 if args.mode == 'concat' else float('inf')   # concat: dtype rounding only; mean: reported, not a failure
    if worst > tol:
        sys.exit(f'average_lora.py: concat self-check failed, worst relative error {worst:.2e} > {tol}')

    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, 'adapter_model.safetensors'), metadata=meta)
    with open(os.path.join(args.out, 'adapter_config.json'), 'w') as f:
        json.dump(new_cfg, f, indent=2)
    with open(os.path.join(args.out, 'omni_average.json'), 'w') as f:
        json.dump({'mode': args.mode, 'checkpoints': [os.path.abspath(c) for c in args.ckpts],
                   'r': new_cfg['r'], 'lora_alpha': new_cfg['lora_alpha'],
                   'self_check_worst_relative_error': worst}, f, indent=2)
    print(f'[average_lora] {args.mode} of {k} -> {args.out} (r={new_cfg["r"]} alpha={new_cfg["lora_alpha"]}, '
          f'self-check worst relative error {worst:.2e})', flush=True)


if __name__ == '__main__':
    main()
