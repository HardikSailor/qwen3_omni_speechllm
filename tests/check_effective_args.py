"""Did the trainer really use the config? Compares three layers for every key of a config YAML:
  config      what the YAML says (after ${VAR} expansion)
  args.json   what ms-swift parsed and post-processed (it writes <run>/args.json)
  runtime     what the live optimizer / scheduler / LoRA layers hold (<run>/omni_effective.json, written by train_omni.py)

    python tests/check_effective_args.py configs/lora_ddp.yaml outputs/<run>/v0-xxx [--expect key ...]

Status per key: OK (all layers agree), OVERRIDDEN (the run changed it on purpose: pass those keys with --expect),
DIFF (unexpected: the trainer changed or ignored the setting), `-` = layer has no such value.
Exit code 1 if any DIFF. Runs anywhere with PyYAML (no GPU, no model)."""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def norm(v):
    """Comparable form: numbers as floats, relative existing paths as real paths, lists element-wise."""
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    if s.startswith('<') and '.' in s and ':' in s:          # enum repr in args.json, e.g. <AttnBackend.flash: 1>
        s = s[1:].split(':')[0].split('.')[-1]
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    try:
        return float(s)
    except ValueError:
        pass
    p = s if os.path.isabs(s) else os.path.join(REPO, s)
    return os.path.realpath(p) if os.path.exists(p) else s


def same(a, b):
    a, b = norm(a), norm(b)
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, list) and len(a) == 1:
        return same(a[0], b)
    if isinstance(b, list) and len(b) == 1:
        return same(a, b[0])
    return a == b


def only(counter_dict):
    """{value: count} with one value -> that value; else the dict (mixed values show up as DIFF)."""
    keys = list(counter_dict)
    return (float(keys[0]) if keys[0] not in (None, 'None') else None) if len(keys) == 1 else counter_dict


def decayed(groups, key='weight_decay'):
    """Weight decay of the groups that decay at all (HF puts biases / norms in a 0.0 group on purpose)."""
    vals = {g[key] for g in groups if g.get(key)}
    return vals.pop() if len(vals) == 1 else sorted(vals) or 0.0


def runtime_values(eff):
    """Map config keys to the values read back from the live trainer objects."""
    lora = eff.get('lora', {})
    r = {'lora_rank': only(lora.get('rank', {})) if lora.get('rank') else None,
         'lora_alpha': only(lora.get('alpha', {})) if lora.get('alpha') else None,
         'lora_dropout': only(lora.get('dropout', {})) if lora.get('dropout') else None}
    if eff.get('trainer', '').startswith('swift'):
        g = eff['optimizer_param_groups']
        lrs = {x.get('initial_lr', x.get('lr')) for x in g}
        r.update({
            'learning_rate': lrs.pop() if len(lrs) == 1 else sorted(lrs),
            'weight_decay': decayed(g),
            'adam_beta2': g[0]['betas'][1] if 'betas' in g[0] else None,
            'lr_scheduler_type': eff['lr_scheduler_type'].split('.')[-1].lower(),
            'warmup_ratio': eff['warmup_steps'] / eff['max_steps'] if eff['max_steps'] else None,
            'max_steps': eff['max_steps'], 'max_grad_norm': eff['max_grad_norm'],
            'per_device_train_batch_size': eff['per_device_train_batch_size'],
            'gradient_accumulation_steps': eff['gradient_accumulation_steps'],
            'gradient_checkpointing': eff['gradient_checkpointing'],
            'seed': eff['seed'], 'data_seed': eff['data_seed']})
    else:
        c, s = eff['optimizer_config'], eff['scheduler']
        r.update({
            'optimizer': c['optimizer'], 'lr': s['max_lr'], 'min_lr': s['min_lr'],
            'weight_decay': s['start_wd'] if s['start_wd'] == s['end_wd'] else [s['start_wd'], s['end_wd']],
            'adam_beta1': c['adam_beta1'], 'adam_beta2': c['adam_beta2'], 'clip_grad': c['clip_grad'],
            'lr_decay_style': s['lr_decay_style'],
            'lr_warmup_fraction': s['lr_warmup_steps'] / s['lr_decay_steps'] if s['lr_decay_steps'] else None,
            'train_iters': eff['train_iters'], 'micro_batch_size': eff['micro_batch_size'],
            'global_batch_size': eff['global_batch_size'],
            'expert_model_parallel_size': eff['expert_model_parallel_size'],
            'recompute_granularity': eff['recompute_granularity'], 'seed': eff['seed']})
    return {k: v for k, v in r.items() if v is not None}


def audio_extended(want, got, cfg):
    """train_omni.py appends audio_tower.layers.N.* names to target_modules when audio_lora_layers is set."""
    want = want if isinstance(want, list) else [want]
    extra = got[len(want):]
    return (cfg.get('audio_lora_layers', ['none'])[0] != 'none' and got[:len(want)] == want and extra
            and all(x.startswith('audio_tower.layers.') for x in extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('run_dir', help='the versioned run folder holding args.json and omni_effective.json')
    ap.add_argument('--expect', nargs='*', default=[], help='keys the run overrode on purpose')
    a = ap.parse_args()
    from train_omni import expand_config
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        argv = expand_config(['--config', a.config])
    cfg, key = {}, None
    for t in argv:
        if t.startswith('--'):
            key = t[2:]; cfg[key] = []
        else:
            cfg[key].append(t)
    args = json.load(open(os.path.join(a.run_dir, 'args.json')))
    eff_path = os.path.join(a.run_dir, 'omni_effective.json')
    eff = json.load(open(eff_path)) if os.path.exists(eff_path) else {}
    rt = runtime_values(eff) if eff else {}
    omni_only = {'mix', 'val_choose', 'audio_lora_layers', 'audio_lora_modules'}   # ours, not ms-swift arguments

    rows, bad = [], 0
    for k, v in cfg.items():
        want = v if len(v) != 1 else v[0]
        layers = [('args.json', args[k]) if k in args else None, ('runtime', rt[k]) if k in rt else None]
        diffs = [n for n, x in filter(None, layers) if not same(want, x)]
        if k in omni_only and k not in args:
            status = 'OK (train_omni option)'
        elif not any(layers):
            status = 'NOT FOUND'
            bad += 1
        elif not diffs:
            status = 'OK'
        elif k in a.expect:
            status = 'OVERRIDDEN'
        elif k == 'target_modules' and diffs == ['args.json'] and audio_extended(want, args[k], cfg):
            status = f'OK + {len(args[k]) - len(want)} audio-encoder names (train_omni)'
        else:
            status = 'DIFF ' + ','.join(diffs)
            bad += 1
        show = lambda x: '-' if x is None else (f'[{len(x)} items]' if isinstance(x, list) and len(x) > 4 else x)
        rows.append((k, show(want), show(args.get(k)), show(rt.get(k)), status))
    w = [max(len(str(r[i])) for r in rows + [('key', 'config', 'args.json', 'runtime', 'status')]) for i in range(5)]
    w = [min(x, 48) for x in w]
    for r in [('key', 'config', 'args.json', 'runtime', 'status')] + rows:
        print('  '.join(str(c)[:48].ljust(w[i]) for i, c in enumerate(r)))

    print(f'\nruntime extras ({eff.get("trainer", "no omni_effective.json")}):')
    if eff:
        lora = eff['lora']
        print(f'  LoRA modules {lora["lora_modules"]} (under audio_tower, incl. proj1/proj2: {lora["audio_encoder_lora_modules"]}; encoder '
              f'layers {lora["audio_encoder_lora_layers"]}); scaling {lora["scaling"]}; '
              f'trainable params (this rank) {lora["trainable_params_this_rank"]:,}')
        for g in eff['optimizer_param_groups']:
            print('  optimizer group:', {k: v for k, v in g.items()})
        for k in ('global_batch_size', 'world_size', 'data_parallel_size', 'warmup_steps', 'scheduler'):
            if k in eff:
                print(f'  {k}: {eff[k]}')
    added = sorted(set(args) & {'dataset', 'val_dataset', 'save_total_limit', 'merge_lora'} - set(cfg))
    if added:
        print('\nset by train_omni.py, not in the config:', {k: args[k] for k in added})
    print(f'\n{"FAIL" if bad else "PASS"}: {bad} unexpected difference(s)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
