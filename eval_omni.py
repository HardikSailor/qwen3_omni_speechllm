"""Checkpoint evaluation for Qwen3-Omni LoRA runs, AudioBench style (as MERaLiON-3 was benchmarked).

    # 1. generate: one process per GPU, each holds the whole model (bf16 ~63 GB) and takes every WORLD_SIZE-th sample;
    #    the base model is loaded once, the checkpoints' LoRA adapters are switched in turn ("base" = no adapter)
    $RUN python -m torch.distributed.run --nproc_per_node 8 eval_omni.py generate --suite evals/suite_v0.yaml \
        --tier quick --models base <run>/checkpoint-5000 <run>/checkpoint-6000 --out_root outputs/evals/<run>/quick
    # 2. score: WER / BLEU / METEOR on CPU; --judge also scores the Llama-3-70B-judge sets (needs the vLLM judge
    #    server on localhost:$MY_VLLM_PORT_JUDGE, see pbs/eval_omni.pbs)
    $RUN python eval_omni.py score --suite evals/suite_v0.yaml --tier quick --out outputs/evals/<run>/checkpoint-5000 [--judge]
    # 3. report: one table, test sets x models
    python eval_omni.py report outputs/evals/<run>/base outputs/evals/<run>/checkpoint-* --out outputs/evals/<run>/report

Generation uses ms-swift's TransformersEngine with the training template (no system prompt, "<audio>" + instruction
in the user turn), greedy decoding, and the PEFT adapter as saved by train_omni.py (vLLM 0.17.1 cannot load
Qwen3-Omni adapters: DEPLOYMENT.md §1). Test sets, prompts and metrics: omni_eval/bench.py and the suite YAML.

Output per model dir: <set>.json (AudioBench format: text, answer, task_type, model_prediction, idx),
<set>_<metric>_score.json, scores.json (headline numbers), meta.json (model, adapter, suite, tier, settings).
Everything is resumable: finished sets are skipped, a killed generate resumes from the per-rank .jsonl files.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from omni_eval import bench  # noqa: E402


def log(msg):
    print(f'[eval_omni {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return None


# ---------------------------------------------------------------- generate

class AdapterSwitcher:
    """One base model per process; checkpoints are switched by unloading the previous LoRA (PEFT `unload()`: the
    original Linear modules come back, base weights untouched, no merge) and wrapping the base again with
    `PeftModel.from_pretrained`, so every checkpoint sees exactly what a fresh load gives, but the 63 GB base is read
    once per job. Not `load_adapter` + `delete_adapter` on the wrapped model: that gave different outputs from a fresh
    load (169 / 552 smoke predictions, job 215360). `None` = the base model."""

    def __init__(self, engine):
        self.engine, self.active = engine, None

    def set(self, path):
        from peft import PeftModel
        m = self.engine.model
        if isinstance(m, PeftModel):
            m = m.unload()
        if path is not None:
            m = PeftModel.from_pretrained(m, path, is_trainable=False)
        m.eval()
        self.engine.model = self.engine.engine = m
        self.active = path


def model_out_name(model):
    return 'base' if model in ('base', 'none') else os.path.basename(os.path.normpath(model))


def make_batches(items, batch_size, max_batch_audio_s):
    """items sorted longest first; at most batch_size clips and max_batch_audio_s seconds of audio per batch
    (a clip longer than the cap goes alone): long recordings don't run out of KV-cache memory."""
    batches, cur, dur = [], [], 0.0
    for it in items:
        if cur and (len(cur) >= batch_size or dur + it['duration'] > max_batch_audio_s):
            batches.append(cur)
            cur, dur = [], 0.0
        cur.append(it)
        dur += it['duration']
    if cur:
        batches.append(cur)
    return batches


def cmd_generate(args):
    import torch
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group('gloo', timeout=datetime.timedelta(hours=6))
        barrier = dist.barrier
    else:
        barrier = lambda: None  # noqa: E731
    torch.cuda.set_device(local_rank)
    sets = bench.load_suite(args.suite, args.tier, args.only)
    plan = []  # (model, out dir, sets to do)
    for model in args.models:
        out = os.path.join(args.out_root, model_out_name(model))
        todo = [s for s in sets if args.overwrite or not os.path.exists(os.path.join(out, f'{s.name}.json'))]
        if todo:
            plan.append((model, out, todo))
    log(f'rank {rank}/{world}: {sum(len(t) for _, _, t in plan)} (model, set) pairs to generate for '
        f'{len(plan)} of {len(args.models)} models -> {args.out_root}')
    if not plan:
        return

    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    t0 = time.time()
    engine = TransformersEngine(args.model, model_type='qwen3_omni_moe', torch_dtype=torch.bfloat16,
                                attn_impl=args.attn_impl, experts_impl='grouped_mm', device_map={'': local_rank},
                                max_batch_size=0)  # batching is done here (make_batches)
    switcher = AdapterSwitcher(engine)
    log(f'rank {rank}: base model loaded in {time.time() - t0:.0f} s')

    for model, out, todo in plan:
        adapter = None if model in ('base', 'none') else os.path.abspath(model)
        t0 = time.time()
        switcher.set(adapter)
        log(f'rank {rank}: adapter {adapter} active ({time.time() - t0:.0f} s)')
        os.makedirs(out, exist_ok=True)
        if rank == 0:
            meta = {'model': args.model, 'adapter': adapter, 'suite': os.path.abspath(args.suite), 'tier': args.tier,
                    'batch_size': args.batch_size, 'max_batch_audio_s': args.max_batch_audio_s, 'decoding': 'greedy',
                    'world_size': world, 'adapter_loading': 'peft load_adapter on a shared base', 'git': git_commit(),
                    'started': time.strftime('%Y-%m-%d %H:%M:%S')}
            with open(os.path.join(out, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
        for s in todo:
            generate_set(s, out, engine, rank, world, barrier, args, InferRequest, RequestConfig)


def generate_set(s, out, engine, rank, world, barrier, args, InferRequest, RequestConfig):
    final = os.path.join(out, f'{s.name}.json')
    part = os.path.join(out, f'{s.name}.rank{rank}.jsonl')
    t0 = time.time()
    items = bench.build_items(s)
    # deal the clips out longest first, so every rank gets a similar amount of long audio
    mine = sorted(items, key=lambda it: (-it['duration'], it['idx']))[rank::world]
    done = set()
    if os.path.exists(part) and not args.overwrite:
        with open(part) as f:
            done = {(r['idx'], r.get('chunk', 0)) for r in map(json.loads, filter(str.strip, f))}
    units = []  # clips, or chunks of the clips longer than s.chunk_s
    for it in mine:
        if s.chunk_s and it['duration'] > s.chunk_s:
            chunks = bench.split_audio(it['audio'], s.chunk_s)
            for k, c in enumerate(chunks):
                units.append({**it, 'audio': c, 'chunk': k, 'n_chunks': len(chunks),
                              'duration': bench.audio_duration(c), 'clip_duration': it['duration']})
        else:
            units.append(it)
    units = [u for u in units if (u['idx'], u.get('chunk', 0)) not in done]
    units.sort(key=lambda u: -u['duration'])
    n_tok = 0
    with open(part, 'w' if args.overwrite else 'a') as f:
        for batch in make_batches(units, args.batch_size, args.max_batch_audio_s):
            cfg = RequestConfig(max_tokens=s.token_budget(max(it['duration'] for it in batch)), temperature=0.0)
            reqs = [InferRequest(messages=[{'role': 'user', 'content': '<audio>' + it['text']}], audios=[it['audio']])
                    for it in batch]
            for it, r in zip(batch, engine.infer(reqs, cfg, use_tqdm=False)):
                rec = {k: v for k, v in it.items() if k != 'audio'}
                rec['model_prediction'] = r.choices[0].message.content
                rec['finish_reason'] = r.choices[0].finish_reason
                n_tok += r.usage.completion_tokens if r.usage else 0
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
    log(f'rank {rank}: {s.name} {len(mine)} samples ({len(units)} requests), '
        f'{sum(it["duration"] for it in mine) / 60:.1f} min audio, {n_tok} tokens in {time.time() - t0:.0f} s')
    barrier()
    if rank == 0:
        parts = {}  # idx -> {chunk: record}
        for p in sorted(glob.glob(os.path.join(out, f'{s.name}.rank*.jsonl'))):
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        parts.setdefault(r['idx'], {})[r.get('chunk', 0)] = r
        recs = {}
        for idx, ch in parts.items():
            if len(ch) == 1 and 'n_chunks' not in ch[0]:
                recs[idx] = ch[0]
                continue
            n = ch[0]['n_chunks']
            if sorted(ch) != list(range(n)):
                raise RuntimeError(f'{s.name}: sample {idx} has chunks {sorted(ch)} of {n}')
            r = {k: v for k, v in ch[0].items() if k not in ('chunk', 'clip_duration')}
            r['duration'] = ch[0]['clip_duration']
            r['model_prediction'] = ' '.join(ch[k]['model_prediction'].strip() for k in range(n))
            r['finish_reason'] = 'length' if any(c['finish_reason'] == 'length' for c in ch.values()) else 'stop'
            recs[idx] = r
        if len(recs) != len(items):
            raise RuntimeError(f'{s.name}: {len(recs)} predictions for {len(items)} samples')
        with open(final + '.tmp', 'w') as f:
            json.dump([recs[i] for i in sorted(recs)], f, indent=1, ensure_ascii=False)
        os.replace(final + '.tmp', final)
        for p in glob.glob(os.path.join(out, f'{s.name}.rank*.jsonl')):
            os.remove(p)
        n_cut = sum(r.get('finish_reason') == 'length' for r in recs.values())
        log(f'{s.name}: {len(recs)} predictions -> {final}' + (f' ({n_cut} hit max_tokens)' if n_cut else ''))
    barrier()


# ---------------------------------------------------------------- score

def cmd_score(args):
    sets = bench.load_suite(args.suite, args.tier, args.only)
    shard = None
    if args.shard:  # i/N: only every N-th set; parallel judge workers, each with its own judge server
        i, n = map(int, args.shard.split('/'))
        sets, shard = sets[i::n], args.shard
    summary_path = os.path.join(args.out, 'scores.json')
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for s in sets:
        pred = os.path.join(args.out, f'{s.name}.json')
        if not os.path.exists(pred):
            log(f'{s.name}: no predictions, skipped')
            continue
        records = json.load(open(pred))
        for metric in s.metrics:
            if metric in bench.JUDGE_METRICS and not args.judge:
                continue
            out = os.path.join(args.out, f'{s.name}_{metric}_score.json')
            if os.path.exists(out) and not args.overwrite:
                result = json.load(open(out))
            else:
                t0 = time.time()
                result = bench.score(s, records, metric)
                with open(out, 'w') as f:
                    json.dump(result, f, indent=1, ensure_ascii=False)
                log(f'{s.name} {metric}: {bench.headline(result, metric):.4f} ({time.time() - t0:.0f} s)')
            summary.setdefault(s.name, {})[metric] = {'value': bench.headline(result, metric), 'n': len(records),
                                                      'task': s.task}
    if shard:  # shards run at once: a final unsharded `score` (all cached by then) writes scores.json
        log(f'shard {shard}: done')
        return
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=1)
    log(f'{len(summary)} sets -> {summary_path}')


# ---------------------------------------------------------------- report

def cmd_report(args):
    def order(d):  # base first, then checkpoints by step, then anything else (e.g. averages) by name
        name = os.path.basename(os.path.normpath(d))
        m = re.fullmatch(r'checkpoint-(\d+)', name)
        return (0, 0, name) if name == 'base' else (1, int(m.group(1)), name) if m else (2, 0, name)

    runs = sorted((d for d in args.dirs if os.path.exists(os.path.join(d, 'scores.json'))), key=order)
    if not runs:
        raise SystemExit('no scores.json in the given dirs')
    names = [os.path.basename(os.path.normpath(d)) for d in runs]
    scores = [json.load(open(os.path.join(d, 'scores.json'))) for d in runs]
    rows = {}  # (task, set, metric) -> [value per run]
    for j, sc in enumerate(scores):
        for name, metrics in sc.items():
            for metric, v in metrics.items():
                rows.setdefault((v['task'], name, metric), [None] * len(runs))[j] = v['value']

    def fmt(v, metric):
        if v is None:
            return '-'
        return f'{100 * v:.2f}' if metric in bench.RATE_METRICS else f'{v:.2f}'

    lines = ['| task | set | metric | ' + ' | '.join(names) + ' |', '|' + '---|' * (3 + len(runs))]
    csv = ['task,set,metric,' + ','.join(names)]
    for (task, name, metric) in sorted(rows):
        vals = rows[(task, name, metric)]
        lines.append(f'| {task} | {name} | {metric} | ' + ' | '.join(fmt(v, metric) for v in vals) + ' |')
        csv.append(f'{task},{name},{metric},' + ','.join('' if v is None else f'{v:.6g}' for v in vals))
    # task averages over the sets every run has (WER / CER in %, others as is)
    lines += ['', '| task | metric | sets | ' + ' | '.join(names) + ' |', '|' + '---|' * (3 + len(runs))]
    groups = {}
    for (task, name, metric), vals in rows.items():
        if all(v is not None for v in vals):
            groups.setdefault((task, metric), []).append(vals)
    for (task, metric), vs in sorted(groups.items()):
        avg = [sum(v[j] for v in vs) / len(vs) for j in range(len(runs))]
        better = 'lower' if not bench.HIGHER_IS_BETTER.get(metric, True) else 'higher'
        lines.append(f'| {task} | {metric} ({better} is better) | {len(vs)} | '
                     + ' | '.join(fmt(a, metric) for a in avg) + ' |')
    text = '\n'.join(lines) + '\n'
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out + '.md', 'w').write(text)
        open(args.out + '.csv', 'w').write('\n'.join(csv) + '\n')
        log(f'-> {args.out}.md, {args.out}.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('generate', 'score'):
        p = sub.add_parser(name)
        p.add_argument('--suite', default=os.path.join(HERE, 'evals', 'suite_v0.yaml'))
        p.add_argument('--tier', default='quick')
        p.add_argument('--only', nargs='*', help='only these set names')
        p.add_argument('--overwrite', action='store_true')
    sub.choices['generate'].add_argument('--out_root', required=True, help='parent of the per-model output dirs')
    sub.choices['score'].add_argument('--out', required=True, help='output dir of one model / checkpoint')
    g = sub.choices['generate']
    g.add_argument('--model', default=os.environ.get('MODEL'), help='base model dir (default $MODEL)')
    g.add_argument('--models', nargs='+', required=True,
                   help='"base" and / or checkpoint dirs with LoRA adapters; the base model is loaded once and the '
                        'adapters are switched; outputs go to <out_root>/<base | checkpoint dir name>')
    g.add_argument('--batch_size', type=int, default=8)
    g.add_argument('--max_batch_audio_s', type=float, default=480, help='cap on audio seconds per batch')
    g.add_argument('--attn_impl', default='flash_attn')
    sub.choices['score'].add_argument('--judge', action='store_true', help='also run the Llama-3-70B judge metrics')
    sub.choices['score'].add_argument('--shard', help='i/N: score only sets i, i+N, ... and leave scores.json alone')
    r = sub.add_parser('report')
    r.add_argument('dirs', nargs='+')
    r.add_argument('--out', help='write <out>.md and <out>.csv')
    args = ap.parse_args()
    {'generate': cmd_generate, 'score': cmd_score, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
