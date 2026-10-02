"""Qwen3-Omni training entry: ms-swift (unmodified) + our MDS data layer (omni_mds, mosaic streaming reader).

    python train_omni.py sft      --mix mixes/st_v0.yaml [omni options] <ordinary `swift sft` arguments>
    python train_omni.py megatron --mix mixes/st_v0.yaml [omni options] <ordinary `megatron sft` arguments>
    python train_omni.py sft --config configs/lora_ddp.yaml [flags that replace the file's values]
With NPROC_PER_NODE set it re-launches itself under torchrun, like the swift / megatron CLIs.

Inside the container (PIPELINE_PLAN.md §3.2, §10):
    source nscc/env.sh      # PYTHONPATH (this dir + container/pydeps), MODEL, MIX, RUN
    $RUN python train_omni.py sft --mix mixes/st_v0.yaml --model <path> --tuner_type lora ...

What this changes compared with plain `swift sft --dataset ...`:
  - datasets: `_prepare_dataset` returns OmniStreamingDataset (train/validation from the mix YAML); each sample is
    turned into a row by omni_mds.sea_text.build_row and encoded by the ms-swift template inside the dataloader
    workers. ms-swift's own dataset loading, preprocessing and LazyLLMDataset are not used;
  - dataloaders: each rank reads its own mosaic partition (no rank-0 dispatcher, no random index access);
  - checkpoints: the mosaic dataloader state is saved as `omni_mosaic_state.json` next to each checkpoint and
    restored on `--resume_from_checkpoint`, with HF's batch skipping turned off (it would re-read and re-encode
    every skipped sample);
  - `OMNI_GPU_MEM_GB=80` caps GPU memory per process to emulate an 80 GB H100 on our H200s (§9.3);
  - `--audio_lora_layers all|top<K>|<a>-<b>` (+ `--audio_lora_modules attn|mlp|attn_mlp`) adds LoRA inside the audio
    encoder: the layer linears are appended to --target_modules (both trainers);
  - `--adapters <checkpoint dir>` (swift sft) starts training from an earlier run's LoRA (weights only: new optimizer,
    LR schedule, data order and W&B run; see check_init_adapters). `--resume_from_checkpoint` is the exact restart;
  - `--wandb true` logs to Weights & Biases through each trainer's own callback (see setup_wandb): one run per launch,
    named after --output_dir, with the run id saved in every checkpoint so a restart continues the same run.
Everything else (model loading, LoRA, template, collator, trainer, saving) is stock ms-swift.
"""
import argparse
import functools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MOSAIC_STATE_FILE = 'omni_mosaic_state.json'
WANDB_STATE_FILE = 'omni_wandb.json'
# ms-swift's SftArguments require --dataset, and switch evaluation off unless --val_dataset is set. Our
# _prepare_dataset ignores both, so the entry passes these placeholders (never loaded).
TRAIN_PLACEHOLDER, VAL_PLACEHOLDER = 'OMNI_MDS_MIX_TRAIN', 'OMNI_MDS_MIX_VALIDATION'
# Linears of one Qwen3-Omni audio encoder layer (thinker.audio_tower.layers.N; 32 layers, d_model 1280).
AUDIO_LORA_MODULES = {'attn': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj'],
                      'mlp': ['fc1', 'fc2']}
AUDIO_LORA_MODULES['attn_mlp'] = AUDIO_LORA_MODULES['attn'] + AUDIO_LORA_MODULES['mlp']


# ------------------------------------------------------------------------------------------------ options

def parse_omni_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--mix', required=True, help='mix YAML (train/validation), see omni_mds/mix.py')
    p.add_argument('--omni_seed', type=int, default=None, help='row/prompt seed (default: --data_seed or 42)')
    # RowConfig (omni_mds/sea_text.py)
    p.add_argument('--asr_prompt_type', default='random')
    p.add_argument('--ac_prompt_type', default='random')
    p.add_argument('--english_prompt_weight', type=float, default=0.5)
    p.add_argument('--augment_prompt', type=_bool, default=False)
    p.add_argument('--asr_normalize_text', type=_bool, default=False)
    p.add_argument('--instruction_dropout_rate', type=float, default=0.0)
    p.add_argument('--text_instructions_path', default=None)
    p.add_argument('--dropout_keep_audio', type=_bool, default=True)
    p.add_argument('--augment_audio', type=_bool, default=False)
    # StreamConfig (omni_mds/mosaic_stream.py)
    p.add_argument('--omni_cache_root', default=None, help='node-local cache (default: $OMNI_MDS_CACHE or /tmp/$USER/...)')
    p.add_argument('--cache_limit', default='600gb')
    p.add_argument('--shuffle_block_size', type=int, default=1_500_000)
    p.add_argument('--num_canonical_nodes', type=int, default=None)
    p.add_argument('--val_choose', type=int, default=None, help='override `choose` of every validation dataset')
    p.add_argument('--clean_stale_shm', type=_bool, default=False,
                   help='remove leaked mosaic shared memory first (only if the job has the node to itself)')
    # LoRA inside the audio encoder (added to --target_modules, see add_audio_lora_targets)
    p.add_argument('--audio_lora_layers', default='none',
                   help='audio encoder layers that get LoRA: none | all | top<K> (last K) | bottom<K> | '
                        '<a>-<b> | i,j,k (0-based, ranges inclusive)')
    p.add_argument('--audio_lora_modules', default='attn', choices=list(AUDIO_LORA_MODULES),
                   help='linears per audio layer: attn = q/k/v/out_proj, mlp = fc1/fc2, attn_mlp = both')
    # Weights & Biases (see setup_wandb). Defaults come from the environment, as in the old shell scripts (WANDB=1 ...).
    env = os.environ.get
    p.add_argument('--wandb', type=_bool, default=_bool(env('WANDB', '0')), help='log to W&B (default: $WANDB, else off)')
    p.add_argument('--wandb_project', default=env('WANDB_PROJECT') or 'meralion_v4')
    p.add_argument('--wandb_entity', default=env('WANDB_ENTITY') or 'i2r-llm', help='W&B team or user')
    p.add_argument('--wandb_run_name', default=env('WANDB_NAME') or None, help='default: basename of --output_dir')
    p.add_argument('--wandb_group', default=env('WANDB_RUN_GROUP') or None,
                   help='groups related runs, e.g. the phases of one PBS job (default: none, or the resumed run\'s)')
    p.add_argument('--wandb_tags', default=env('WANDB_TAGS') or None, help='comma separated')
    p.add_argument('--wandb_mode', default=env('WANDB_MODE') or None, choices=[None, 'online', 'offline', 'disabled'],
                   help='offline: write to <output_dir>/wandb and upload later with `wandb sync`')
    p.add_argument('--wandb_dir', default=None, help='where the wandb/ folder goes (default: --output_dir)')
    p.add_argument('--wandb_resume', default='auto', choices=['auto', 'always', 'never'],
                   help='on resume, continue the checkpoint\'s W&B run: auto = only when the checkpoint is inside '
                        '--output_dir (a restart of the same run), always, never (new run in the same group)')
    return p.parse_known_args(argv)


def expand_config(argv):
    """`--config run.yaml` -> the flags it lists, placed before the command-line ones.

    The YAML maps flag names (without `--`) to values; both our options (mix, val_choose, audio_lora_layers, ...) and any
    `swift sft` / `megatron sft` argument are allowed. Values: lists become several tokens, booleans `true`/`false`,
    dicts JSON, `null` drops the key; `$VAR` / `${VAR}` are expanded from the environment (e.g. `model: ${MODEL}`).
    `target_modules_file: <file>` reads the names for --target_modules from a file (relative to this repo).
    A flag given on the command line replaces the config's value for that flag (the config entry is dropped, not merged)."""
    idx = [i for i, x in enumerate(argv) if x == '--config' or x.startswith('--config=')]
    if not idx:
        return argv
    if len(idx) > 1:
        sys.exit('train_omni.py: give --config once')
    i = idx[0]
    if argv[i] == '--config':
        if i + 1 >= len(argv):
            sys.exit('train_omni.py: --config needs a file')
        path, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    else:
        path, argv = argv[i].split('=', 1)[1], argv[:i] + argv[i + 1:]
    import yaml
    _RUN_FILES['config'] = os.path.abspath(path)
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        sys.exit(f'train_omni.py: {path} must be a mapping of flag: value')
    if 'target_modules_file' in cfg:
        if 'target_modules' in cfg:
            sys.exit(f'train_omni.py: {path} has both target_modules and target_modules_file')
        f = os.path.expandvars(str(cfg.pop('target_modules_file')))
        f = f if os.path.isabs(f) else os.path.join(HERE, f)
        with open(f, encoding='utf-8') as fh:
            cfg['target_modules'] = fh.read().split()

    def tokens(key, v):
        if isinstance(v, bool):
            return ['true' if v else 'false']
        if isinstance(v, (list, tuple)):
            return [t for x in v for t in tokens(key, x)]
        if isinstance(v, dict):
            return [json.dumps(v)]
        s = os.path.expandvars(str(v))
        if '$' in s:
            sys.exit(f'train_omni.py: {path}: {key}: unset environment variable in {v!r}')
        return [s]

    on_cli = {x[2:].split('=')[0] for x in argv if x.startswith('--')}
    out, replaced = [], []
    for key, v in cfg.items():
        if key in on_cli:
            replaced.append(key)
        elif v is not None:
            out += [f'--{key}'] + tokens(key, v)
    print(f'[train_omni] config {path}: {len(cfg)} settings'
          + (f'; replaced on the command line: {" ".join(replaced)}' if replaced else ''), flush=True)
    return out + argv


_RUN_FILES = {}   # files that define the run (--config, --mix), copied into <output_dir>/omni_run by save_run_files


def save_run_files(omni, swift_argv, raw_argv):
    """Rank 0 keeps what defined the run next to its results: <output_dir>/omni_run/ holds the --config YAML, the mix YAML,
    the command line as given and as passed to ms-swift, and $OMNI_RUN_EXTRA_FILES (e.g. the PBS script). A resumed run
    adds a new numbered folder instead of overwriting (omni_run, omni_run.1, ...)."""
    import shutil
    import time
    if os.environ.get('RANK', '0') != '0':
        return
    out = argv_value(swift_argv, '--output_dir')
    if not out:
        return
    d = os.path.join(out, 'omni_run')
    k = 0
    while os.path.exists(d):
        k += 1
        d = os.path.join(out, f'omni_run.{k}')
    os.makedirs(d)
    files = dict(_RUN_FILES, mix=os.path.abspath(omni.mix))
    for extra in filter(None, os.environ.get('OMNI_RUN_EXTRA_FILES', '').split(',')):
        files[os.path.basename(extra)] = os.path.abspath(extra)
    for key, src in files.items():
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(d, os.path.basename(src) if key not in ('config', 'mix') else
                                           f'{key}__{os.path.basename(src)}'))
    with open(os.path.join(d, 'command.json'), 'w') as f:
        json.dump({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'cwd': os.getcwd(), 'argv': raw_argv,
                   'files': files, 'omni': vars(omni), 'swift_argv': swift_argv,
                   'env': {k: v for k, v in os.environ.items()
                           if k.startswith(('OMNI_', 'NNODES', 'NODE_RANK', 'NPROC', 'WORLD_SIZE', 'MASTER_', 'PBS_', 'WANDB_'))
                           and 'KEY' not in k}}, f, indent=1)
    print(f'[train_omni] run files -> {d}', flush=True)


def _bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes'):
        return True
    if v.lower() in ('0', 'false', 'no'):
        return False
    raise argparse.ArgumentTypeError(v)


def apply_gpu_mem_cap():
    """OMNI_GPU_MEM_GB=<GB>: cap this process's CUDA memory to emulate a smaller GPU (e.g. 80 on an H200)."""
    gb = os.environ.get('OMNI_GPU_MEM_GB')
    if not gb:
        return
    import torch
    if not torch.cuda.is_available():
        return
    device = int(os.environ.get('LOCAL_RANK', 0))
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, float(gb) * 1e9 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    print(f'[train_omni] OMNI_GPU_MEM_GB={gb}: GPU {device} capped at {fraction:.3f} of '
          f'{total / 1e9:.0f} GB = {fraction * total / 2**30:.1f} GiB', flush=True)


# ------------------------------------------------------------------------------------------------ encoding

def encode_row(template, row):
    """Row -> encoded sample (raises on errors / too long; the dataset then replaces the sample)."""
    row = {k: v for k, v in row.items() if k != '_meta'}
    return template.encode(row, return_length=True)


# ------------------------------------------------------------------------------------------------ effective settings

EFFECTIVE_FILE = 'omni_effective.json'


def _by_adapter(module, attr, adapter, default=None):
    """module.<attr>[adapter] for dicts and nn.ModuleDict (PEFT keeps lora_dropout in a ModuleDict)."""
    d = getattr(module, attr, None)
    try:
        return d[adapter] if d is not None and adapter in d else default
    except TypeError:
        return default


def lora_summary(models):
    """What the LoRA layers in the live model(s) hold: rank, alpha, scaling, dropout, how many and where."""
    import collections
    ranks, alphas, scalings, dropouts = (collections.Counter() for _ in range(4))
    names, n_train, n_total = [], 0, 0
    for model in models:
        for name, m in model.named_modules():
            r = getattr(m, 'r', None)
            if isinstance(r, dict) and r and hasattr(m, 'lora_A'):
                names.append(name)
                for adapter, rank in r.items():
                    ranks[rank] += 1
                    alphas[_by_adapter(m, 'lora_alpha', adapter)] += 1
                    scalings[round(float(_by_adapter(m, 'scaling', adapter, float('nan'))), 6)] += 1
                    dropouts[getattr(_by_adapter(m, 'lora_dropout', adapter), 'p', 0.0)] += 1   # nn.Identity: 0
        for p in model.parameters():
            n_total += p.numel()
            n_train += p.numel() if p.requires_grad else 0
    audio = [n for n in names if 'audio_tower' in n]
    return {'lora_modules': len(names), 'rank': dict(ranks), 'alpha': dict(alphas), 'scaling': dict(scalings),
            'dropout': dict(dropouts), 'audio_encoder_lora_modules': len(audio),
            'audio_encoder_lora_layers': sorted({int(n.split('audio_tower.layers.')[1].split('.')[0])
                                                 for n in audio if 'audio_tower.layers.' in n}),
            'examples': names[:3], 'trainable_params_this_rank': n_train, 'total_params_this_rank': n_total}


def group_summary(param_groups, keys):
    """Optimizer param groups -> the settings that differ between them, with how many params each group holds."""
    out = []
    for g in param_groups:
        e = {k: (list(g[k]) if isinstance(g[k], tuple) else g[k]) for k in keys if k in g}
        e['num_tensors'] = len(g['params'])
        e['num_params'] = sum(p.numel() for p in g['params'])
        out.append(e)
    return out


def write_effective(output_dir, info):
    """Rank 0 writes what the trainer actually built (optimizer, scheduler, LoRA, batch) to <output_dir>/omni_effective.json,
    read back from the live objects, not from the arguments. tests/check_effective_args.py compares it with a config."""
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, EFFECTIVE_FILE)
    with open(path, 'w') as f:
        json.dump(info, f, indent=1, default=str)
    print(f'[train_omni] effective settings -> {path}', flush=True)


# ------------------------------------------------------------------------------------------------ W&B

_WANDB_EXTRA_CONFIG = {}   # omni options + mix, added to the W&B run's config (set in main)


def wandb_run():
    """The active wandb run in this process, or None (only the trainer's logging rank has one)."""
    wandb = sys.modules.get('wandb')
    return getattr(wandb, 'run', None) if wandb is not None else None


def wandb_update_config(info):
    """Add our settings to the run's config (on the logging rank; a no-op elsewhere or without W&B)."""
    run = wandb_run()
    if run is not None:
        run.config.update({**_WANDB_EXTRA_CONFIG, **info}, allow_val_change=True)


def save_wandb_state(ckpt):
    """The logging rank writes its run id next to the checkpoint, so a resume can continue the same run."""
    run = wandb_run()
    if run is None or not ckpt:
        return
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, WANDB_STATE_FILE), 'w') as f:
        json.dump({'id': run.id, 'name': run.name, 'project': run.project, 'entity': run.entity,
                   'group': run.group, 'url': run.url}, f)


def setup_wandb(mode, omni, swift_argv):
    """--wandb true -> the trainer arguments and WANDB_* variables that make both trainers log to the same project.

    Both trainers log through their own callback on one rank (swift sft: transformers' WandbCallback on rank 0, logging
    every --logging_steps against train/global_step; megatron: Megatron-SWIFT's callback on the last rank, against the
    iteration). Neither passes entity / group / tags / run id to wandb.init, so they go in the environment, which
    wandb.init reads. Every rank calls this with the same arguments, so they all agree.
    Resume: the checkpoint's omni_wandb.json holds the run id. With --wandb_resume auto the run is continued when the
    checkpoint lies inside --output_dir (a restart), else a new run starts in the same group. Megatron logs by
    iteration, so after a restart W&B drops the points between the checkpoint and the last logged iteration."""
    if not omni.wandb:
        return swift_argv
    out = argv_value(swift_argv, '--output_dir')
    if not out:
        sys.exit('train_omni.py: --wandb needs --output_dir (it names the run and holds the wandb/ folder)')
    out = os.path.abspath(out)
    name = omni.wandb_run_name or os.path.basename(out.rstrip('/'))
    group, notes, run_id = omni.wandb_group, None, None

    if mode == 'sft':
        ckpt = argv_value(swift_argv, '--resume_from_checkpoint')
    elif (argv_value(swift_argv, '--finetune') or '').lower() in ('false', '0'):
        ckpt = argv_value(swift_argv, '--mcore_adapter') or argv_value(swift_argv, '--mcore_model')
    else:
        ckpt = None
    prev = None
    if ckpt and os.path.isfile(os.path.join(ckpt, WANDB_STATE_FILE)):
        with open(os.path.join(ckpt, WANDB_STATE_FILE)) as f:
            prev = json.load(f)
    if prev:
        inside = os.path.abspath(ckpt).startswith(out + os.sep)
        if omni.wandb_resume == 'always' or (omni.wandb_resume == 'auto' and inside):
            run_id, name = prev['id'], omni.wandb_run_name or prev.get('name') or name
        else:
            group = group or prev.get('group') or prev.get('name')
            notes = f'resumed from {ckpt} (W&B run {prev.get("name")} / {prev["id"]})'

    wdir = os.path.abspath(omni.wandb_dir or out)
    os.makedirs(wdir, exist_ok=True)
    env = {'WANDB_PROJECT': omni.wandb_project, 'WANDB_ENTITY': omni.wandb_entity, 'WANDB_RUN_GROUP': group,
           'WANDB_TAGS': omni.wandb_tags, 'WANDB_MODE': omni.wandb_mode, 'WANDB_DIR': wdir, 'WANDB_NOTES': notes,
           'WANDB_RUN_ID': run_id, 'WANDB_RESUME': 'allow' if run_id else None,
           'WANDB_LOG_MODEL': os.environ.get('WANDB_LOG_MODEL', 'false')}   # never upload checkpoints by default
    for k in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NOTES', 'WANDB_RUN_GROUP', 'WANDB_NAME'):
        os.environ.pop(k, None)   # not inherited from the shell: this function decides them
    os.environ.update({k: v for k, v in env.items() if v})

    # report_to: add wandb to an explicit list (dropping `none`), else tensorboard (the default) + wandb
    starts = [i for i, x in enumerate(swift_argv) if x == '--report_to']
    if any(x.startswith('--report_to=') for x in swift_argv):
        sys.exit('train_omni.py: with --wandb write `--report_to a b`, not `--report_to=`')
    if not starts:
        swift_argv = swift_argv + ['--report_to', 'tensorboard', 'wandb']
    else:
        i = starts[-1]
        end = i + 1
        while end < len(swift_argv) and not swift_argv[end].startswith('--'):
            end += 1
        kept = [x for x in swift_argv[i + 1:end] if x not in ('none', 'wandb')]   # `none` (the recipes) + wandb is invalid
        swift_argv = swift_argv[:i + 1] + kept + ['wandb'] + swift_argv[end:]
    given = {x.split('=')[0] for x in swift_argv if x.startswith('--')}
    if mode == 'sft':
        swift_argv += [] if '--run_name' in given else ['--run_name', name]
    else:
        swift_argv += [] if '--wandb_project' in given else ['--wandb_project', omni.wandb_project]
        swift_argv += [] if '--wandb_exp_name' in given else ['--wandb_exp_name', name]
    if os.environ.get('RANK', '0') != '0':
        return swift_argv
    print(f'[train_omni] W&B: project {omni.wandb_project}, run {name}'
          + (f' (continuing {run_id})' if run_id else '') + (f', group {group}' if group else '')
          + (f', mode {omni.wandb_mode}' if omni.wandb_mode else '') + f', dir {wdir}', flush=True)
    if not os.environ.get('WANDB_API_KEY') and omni.wandb_mode not in ('offline', 'disabled'):
        print('[train_omni] W&B: WANDB_API_KEY is not set (nscc/env.sh reads it from ~/.netrc); wandb.init may fail',
              flush=True)
    return swift_argv


# ------------------------------------------------------------------------------------------------ trainer

def make_streaming_trainer_cls(base, fix_adapter_config=False):
    """Subclass of the ms-swift trainer class that reads OmniStreamingDataset with per-rank mosaic dataloaders."""
    from swift.utils import get_logger

    from omni_mds.mosaic_stream import OmniStreamingDataset, make_dataloader
    logger = get_logger()

    from transformers import TrainerCallback

    class EffectiveSettingsCallback(TrainerCallback):
        """At train start (optimizer and scheduler exist): record what the HF trainer really uses."""

        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_begin(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
            t = self.trainer
            world = t.accelerator.num_processes if getattr(t, 'accelerator', None) else 1
            info = {
                'trainer': 'swift sft (HF)',
                'optimizer_class': type(getattr(optimizer, 'optimizer', optimizer)).__name__,
                'optimizer_param_groups': group_summary(
                    optimizer.param_groups, ('initial_lr', 'lr', 'weight_decay', 'betas', 'eps')),
                'scheduler_class': type(lr_scheduler).__name__,
                'lr_scheduler_type': str(args.lr_scheduler_type),
                'warmup_steps': args.get_warmup_steps(state.max_steps), 'max_steps': state.max_steps,
                'max_grad_norm': args.max_grad_norm,
                'world_size': world, 'per_device_train_batch_size': args.per_device_train_batch_size,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'global_batch_size': args.per_device_train_batch_size * args.gradient_accumulation_steps * world,
                'gradient_checkpointing': bool(getattr(model, 'is_gradient_checkpointing', args.gradient_checkpointing)),
                'seed': args.seed, 'data_seed': args.data_seed,
                'init_adapter': _INIT_ADAPTER.get('path'),
                'lora': lora_summary([model]),
            }
            write_effective(args.output_dir, info)
            wandb_update_config({'omni_effective': info})

    class OmniStreamingTrainer(base):
        _omni_train_dl = None
        _omni_eval_dls = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(EffectiveSettingsCallback(self))

        def _omni_loader(self, dataset, batch_size, persistent):
            a = self.args
            return make_dataloader(dataset, batch_size=batch_size, collate_fn=self.data_collator,
                                   num_workers=a.dataloader_num_workers, pin_memory=a.dataloader_pin_memory,
                                   prefetch_factor=a.dataloader_prefetch_factor or 2,
                                   persistent_workers=persistent and a.dataloader_num_workers > 0)

        def get_train_dataloader(self, skip_batches=0):
            if not isinstance(self.train_dataset, OmniStreamingDataset):
                return super().get_train_dataloader(skip_batches=skip_batches)
            if skip_batches:
                raise RuntimeError('omni_mds: batch skipping is not supported; resume uses the mosaic state '
                                   f'({MOSAIC_STATE_FILE}) instead')
            dl = self._omni_loader(self.train_dataset, self._train_batch_size,
                                   self.args.dataloader_persistent_workers)
            ckpt = self.args.resume_from_checkpoint
            if isinstance(ckpt, str) and ckpt:
                path = os.path.join(ckpt, MOSAIC_STATE_FILE)
                if os.path.isfile(path):
                    with open(path) as f:
                        state = json.load(f)
                    dl.load_state_dict(state)
                    self.args.ignore_data_skip = True   # the mosaic state already positions the data
                    logger.info(f'omni_mds: resumed data position from {path}: {state}')
                else:
                    logger.warning(f'omni_mds: {path} not found; data restarts at the beginning of the epoch')
                    self.args.ignore_data_skip = True
            self._omni_train_dl = dl
            return dl

        def get_eval_dataloader(self, eval_dataset=None):
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            if not isinstance(ds, OmniStreamingDataset):
                return super().get_eval_dataloader(eval_dataset)
            self._omni_eval_dls = self._omni_eval_dls or {}
            key = id(ds)
            if key not in self._omni_eval_dls:  # reuse: the validation set is the same fixed subset every time
                self._omni_eval_dls[key] = self._omni_loader(ds, self.args.eval_batch_size, persistent=True)
            return self._omni_eval_dls[key]

        def _save_checkpoint(self, *args, **kwargs):
            result = super()._save_checkpoint(*args, **kwargs)
            ckpt = getattr(self.state, 'last_model_checkpoint', None)
            if self._omni_train_dl is not None and ckpt and self.args.should_save:
                state = self._omni_train_dl.state_dict()
                os.makedirs(ckpt, exist_ok=True)
                with open(os.path.join(ckpt, MOSAIC_STATE_FILE), 'w') as f:
                    json.dump(state, f)
                logger.info(f'omni_mds: saved data position {state} to {ckpt}')
            if fix_adapter_config and ckpt and self.args.should_save:
                fix_peft_target_modules(ckpt)
            save_wandb_state(ckpt)
            return result

    OmniStreamingTrainer.__name__ = f'OmniStreaming{base.__name__}'
    return OmniStreamingTrainer


# ------------------------------------------------------------------------------------------------ pipeline

def build_omni_datasets(pipeline, omni, train_batch_size, eval_batch_size, want_val):
    """Train/validation OmniStreamingDataset from the mix YAML; shared by the HF and Megatron pipelines."""
    from swift.utils import get_logger

    from omni_mds.mix import load_mix
    from omni_mds.mosaic_stream import OmniStreamingDataset, StreamConfig
    from omni_mds.sea_text import RowConfig
    a = pipeline.args
    mix = load_mix(omni.mix)
    seed = omni.omni_seed if omni.omni_seed is not None else (a.data_seed if a.data_seed is not None else 42)
    row_cfg = RowConfig(asr_prompt_type=omni.asr_prompt_type, ac_prompt_type=omni.ac_prompt_type,
                        english_prompt_weight=omni.english_prompt_weight, augment_prompt=omni.augment_prompt,
                        asr_normalize_text=omni.asr_normalize_text,
                        instruction_dropout_rate=omni.instruction_dropout_rate,
                        text_instructions_path=omni.text_instructions_path,
                        dropout_keep_audio=omni.dropout_keep_audio)
    common = dict(cache_limit=omni.cache_limit, num_canonical_nodes=omni.num_canonical_nodes)
    if omni.omni_cache_root:
        common['cache_root'] = omni.omni_cache_root
    encode = functools.partial(encode_row, pipeline.template)

    train = OmniStreamingDataset(
        mix['train'], 'train', row_cfg, batch_size=train_batch_size, is_train=True, seed=seed,
        stream_cfg=StreamConfig(shuffle_seed=seed, shuffle_block_size=omni.shuffle_block_size, **common),
        transform=encode, augment=omni.augment_audio)
    val, specs = None, mix['validation']
    if specs and want_val:
        if omni.val_choose is not None:
            specs = [type(s)(**{**s.__dict__, 'choose': omni.val_choose, 'repeat': None}) for s in specs]
        val = OmniStreamingDataset(
            specs, 'validation', row_cfg, batch_size=eval_batch_size, is_train=False,
            seed=seed, stream_cfg=StreamConfig.for_validation(**common), transform=encode)
    get_logger().info(
        f'omni_mds: mix {omni.mix}: train {len(mix["train"])} datasets, {train.epoch_size} samples/epoch '
        f'({len(train)} per rank); validation '
        f'{"none" if val is None else f"{len(specs)} datasets, {val.epoch_size} samples"}')
    return [train, val]


def check_common_args(a):
    bad = [name for name, cond in [('--streaming', getattr(a, 'streaming', False)),
                                   ('--packing', getattr(a, 'packing', False)),
                                   ('--group_by_length', getattr(a, 'group_by_length', False)),
                                   ('--dataset', list(a.dataset) != [TRAIN_PLACEHOLDER]),
                                   ('--val_dataset', list(a.val_dataset) not in ([], [VAL_PLACEHOLDER]))] if cond]
    if bad:
        raise ValueError(f'train_omni.py: not supported with the MDS mix: {bad} (data comes from --mix)')


def make_sft_cls(omni):
    from swift.pipelines import SwiftSft
    from swift.trainers import TrainerFactory

    _orig_get_trainer_cls = TrainerFactory.get_trainer_cls.__func__

    def get_trainer_cls(cls, args):
        return make_streaming_trainer_cls(_orig_get_trainer_cls(cls, args),
                                          fix_adapter_config=audio_lora_enabled(omni))

    TrainerFactory.get_trainer_cls = classmethod(get_trainer_cls)

    class OmniSwiftSft(SwiftSft):

        def _prepare_dataset(self):
            a = self.args
            check_common_args(a)
            if a.sequence_parallel_size > 1:
                raise ValueError('train_omni.py: sequence parallelism is not supported yet')
            return build_omni_datasets(self, omni, a.per_device_train_batch_size, a.per_device_eval_batch_size,
                                       want_val=a.eval_strategy != 'no')

    return OmniSwiftSft


# ------------------------------------------------------------------------------------------------ Megatron

class _GlobalLen:
    """Megatron's args.init_iters() reads len(dataset) as the *global* sample count (it was written for map-style
    datasets). A mosaic dataset's len() is per rank, so hand it the global epoch size instead."""

    def __init__(self, dataset):
        self.n = dataset.epoch_size

    def __len__(self):
        return self.n


def make_omni_megatron_trainer_cls(base, fix_adapter_config=False):
    import torch.distributed as dist
    from megatron.core import mpu
    from swift.utils import get_logger

    from omni_mds.mosaic_stream import OmniStreamingDataset, make_dataloader
    logger = get_logger()

    class OmniMegatronTrainer(base):
        _omni_train_dl = None
        _omni_effective = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # the W&B callback (last rank) is created at the end of the base __init__, after the optimizer
            wandb_update_config({'omni_effective': self._omni_effective or {}})

        def get_optimizer_and_scheduler(self):
            optimizer, sched = super().get_optimizer_and_scheduler()
            a = self.args
            opts = getattr(optimizer, 'chained_optimizers', [optimizer])
            cfg = getattr(opts[0], 'config', None)
            info = {
                'trainer': 'megatron sft (Megatron-SWIFT)',
                'optimizer_class': [type(o).__name__ for o in opts],
                'optimizer_config': {k: getattr(cfg, k, None) for k in (
                    'optimizer', 'lr', 'min_lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps',
                    'clip_grad', 'bf16', 'use_distributed_optimizer')},
                'optimizer_param_groups': group_summary(
                    optimizer.param_groups, ('max_lr', 'min_lr', 'lr', 'weight_decay', 'wd_mult', 'lr_mult',
                                             'is_decoupled_lr')),
                'scheduler': {k: getattr(sched, k, None) for k in (
                    'max_lr', 'min_lr', 'lr_warmup_steps', 'lr_decay_steps', 'lr_decay_style',
                    'start_wd', 'end_wd', 'wd_incr_style')},
                'train_iters': a.train_iters, 'micro_batch_size': a.micro_batch_size,
                'global_batch_size': a.global_batch_size,
                'data_parallel_size': mpu.get_data_parallel_world_size(),
                'expert_model_parallel_size': mpu.get_expert_model_parallel_world_size(),
                'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
                'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
                'recompute_granularity': a.recompute_granularity, 'seed': a.seed,
                'lora': lora_summary(self.wrapped_models),
            }
            write_effective(a.output_dir, info)
            self._omni_effective = info
            return optimizer, sched

        def _omni_loader(self, dataset):
            a = self.args
            workers = a.dataloader_num_workers
            return make_dataloader(dataset, batch_size=a.micro_batch_size, collate_fn=self.data_collator,
                                   num_workers=workers, pin_memory=a.dataloader_pin_memory,
                                   prefetch_factor=getattr(a, 'dataloader_prefetch_factor', None) or 2,
                                   persistent_workers=bool(getattr(a, 'dataloader_persistent_workers', True))
                                   and workers > 0)

        def _resume_dirs(self):
            a = self.args
            if a.finetune:
                return []
            dirs = list(a.adapters or []) + [a.model] + [getattr(a, 'mcore_adapter', None), getattr(a, 'mcore_model', None)]
            return [d for d in dirs if d]

        def _prepare_dataloader(self, train_dataset, val_dataset=None):
            if not isinstance(train_dataset, OmniStreamingDataset):
                return super()._prepare_dataloader(train_dataset, val_dataset)
            world = dist.get_world_size() if dist.is_initialized() else 1
            if mpu.get_data_parallel_world_size() != world:
                raise NotImplementedError(
                    'omni_mds: mosaic partitions by global rank, so every rank must be its own data-parallel rank '
                    f'(TP=PP=CP=1; EP is fine). Got DP={mpu.get_data_parallel_world_size()}, world={world}.')
            dl = self._omni_loader(train_dataset)
            it = self.state.iteration   # already loaded from the checkpoint by the base trainer
            candidates = [p for d in self._resume_dirs()
                          for p in (d, os.path.join(d, f'checkpoint-{it}'), os.path.join(os.path.dirname(d), f'checkpoint-{it}'))]
            for d in candidates:
                path = os.path.join(d, MOSAIC_STATE_FILE)
                if os.path.isfile(path):
                    with open(path) as f:
                        state = json.load(f)
                    dl.load_state_dict(state)
                    logger.info(f'omni_mds: resumed data position from {path}: {state}')
                    break
            else:
                if self._resume_dirs() and self.state.consumed_train_samples:
                    logger.warning('omni_mds: resuming without omni_mosaic_state.json; data restarts at the epoch start')
            self._omni_train_dl = dl
            val_dl = self._omni_loader(val_dataset) if val_dataset is not None else None
            return dl, val_dl

        def save_checkpoint(self):
            result = super().save_checkpoint()
            ckpt = self.state.last_model_checkpoint
            if self._omni_train_dl is not None and ckpt and (not dist.is_initialized() or dist.get_rank() == 0):
                state = self._omni_train_dl.state_dict()
                os.makedirs(ckpt, exist_ok=True)
                with open(os.path.join(ckpt, MOSAIC_STATE_FILE), 'w') as f:
                    json.dump(state, f)
                logger.info(f'omni_mds: saved data position {state} to {ckpt}')
            if fix_adapter_config and ckpt and (not dist.is_initialized() or dist.get_rank() == 0):
                fix_peft_target_modules(ckpt)
            save_wandb_state(ckpt)   # written by the rank that holds the W&B run
            return result

    OmniMegatronTrainer.__name__ = f'Omni{base.__name__}'
    return OmniMegatronTrainer


def _install_optimizer_load_debug():
    """OMNI_DEBUG_OPTIM_LOAD=1: print the structure of the optimizer state dict Megatron-SWIFT hands to
    optimizer.load_state_dict on resume (debugging the mcore 0.16.1 resume failure, PIPELINE_PLAN.md §10.4)."""
    from swift.megatron.utils import megatron_lm_utils as mlu
    orig = mlu._load_optimizer_state_dict

    def describe(obj, depth=0, max_depth=3):
        pad = '  ' * depth
        if isinstance(obj, dict) and depth < max_depth:
            return '\n'.join(f'{pad}{k!r}: {type(v).__name__}' + ('\n' + describe(v, depth + 1, max_depth)
                                                                   if isinstance(v, dict) else '')
                             for k, v in list(obj.items())[:40])
        return f'{pad}{type(obj).__name__}'

    def wrapped(optimizer, state_dict):
        print(f'[omni debug] optimizer {type(optimizer).__name__}; state_dict passed to load_state_dict:\n'
              f'{describe(state_dict)}', flush=True)
        return orig(optimizer, state_dict)

    mlu._load_optimizer_state_dict = wrapped


def make_megatron_sft_cls(omni):
    from swift.megatron.pipelines.train.sft import MegatronSft
    from swift.megatron.trainers import MegatronTrainer
    if os.environ.get('OMNI_DEBUG_OPTIM_LOAD'):
        _install_optimizer_load_debug()

    class OmniMegatronSft(MegatronSft):

        def _prepare_dataset(self):
            a = self.args
            check_common_args(a)
            tp, pp, cp = a.tensor_model_parallel_size, a.pipeline_model_parallel_size, a.context_parallel_size
            if (tp, pp, cp) != (1, 1, 1):
                raise NotImplementedError(f'train_omni.py megatron: TP/PP/CP must be 1 for the mosaic reader '
                                          f'(got {tp}/{pp}/{cp}); EP is supported (PIPELINE_PLAN.md §8.4)')
            want_val = getattr(a, 'eval_iters', -1) != 0
            return build_omni_datasets(self, omni, a.micro_batch_size, a.micro_batch_size, want_val=want_val)

        def run(self):
            orig = self.args.init_iters

            def init_iters(train_dataset, val_dataset):
                return orig(_GlobalLen(train_dataset), _GlobalLen(val_dataset) if val_dataset is not None else None)

            self.args.init_iters = init_iters
            return super().run()

        def prepare_trainer(self):
            if self.args.task_type not in (None, 'causal_lm'):
                raise NotImplementedError(f'train_omni.py megatron: task_type {self.args.task_type}')
            return make_omni_megatron_trainer_cls(MegatronTrainer, fix_adapter_config=audio_lora_enabled(omni))(
                self.args, self.template)

    return OmniMegatronSft


# ------------------------------------------------------------------------------------------------ main

def maybe_relaunch_with_torchrun(argv):
    """Like the swift / megatron CLIs: with NPROC_PER_NODE set, re-run this script under torchrun."""
    if not os.environ.get('NPROC_PER_NODE') or 'LOCAL_RANK' in os.environ:
        return
    import subprocess
    torchrun = []
    for key in ('NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR'):
        if os.environ.get(key):
            torchrun += [f'--{key.lower()}', os.environ[key]]
    cmd = [sys.executable, '-m', 'torch.distributed.run', *torchrun, os.path.abspath(__file__), *argv]
    print(f'[train_omni] run: {" ".join(cmd)}', flush=True)
    sys.exit(subprocess.run(cmd).returncode)


def argv_value(swift_argv, name):
    """Value of a single-valued `--name v` / `--name=v` argument, or None."""
    for i, x in enumerate(swift_argv):
        if x == name and i + 1 < len(swift_argv):
            return swift_argv[i + 1]
        if x.startswith(name + '='):
            return x.split('=', 1)[1]
    return None


def check_megatron_resume_args(swift_argv):
    """Megatron-SWIFT (ms-swift 8ec0455, mcore 0.16.1) builds the checkpoint *load* template with the resuming
    run's args, and `_generate_state_dict` only adds the optimizer entries when `not args.no_save_optim`. So
    resuming with the optimizer while passing `--no_save_optim true` loads an empty optimizer state
    (`KeyError: 'optimizer'` / `KeyError: 'state'`, or NaN gradients with the fully-reshardable format).
    Found 2026-09-28 (PIPELINE_PLAN.md §10.4). Refuse that combination up front."""
    def value(name):
        v = argv_value(swift_argv, name)
        return None if v is None else v.lower()

    resuming = value('--finetune') in ('false', '0') and (value('--mcore_adapter') or value('--mcore_model'))
    loads_optim = value('--no_load_optim') not in ('true', '1')
    if resuming and loads_optim and value('--no_save_optim') in ('true', '1'):
        sys.exit('train_omni.py megatron: resuming with the optimizer needs --no_save_optim false (Megatron-SWIFT '
                 'uses the save flag to build its load template; with true the optimizer state loads empty). '
                 'Either drop --no_save_optim true or pass --no_load_optim true.')


def fix_hf_warmup_steps(mode, swift_argv):
    """swift sft: `--warmup_steps N` is silently dropped. ms-swift (8ec0455) declares `warmup_ratio: float = 0.` and
    transformers 5.2's TrainingArguments.__post_init__ does `if self.warmup_ratio is not None: self.warmup_steps =
    self.warmup_ratio`, so warmup_steps becomes 0 (found 2026-10-02, job 215264: LR 1e-4 at step 1). Pass the same warmup
    as a ratio of --max_steps instead (transformers turns a ratio < 1 back into ceil(max_steps * ratio) steps)."""
    steps, max_steps = argv_value(swift_argv, '--warmup_steps'), argv_value(swift_argv, '--max_steps')
    if mode != 'sft' or steps is None or float(steps) < 1:
        return swift_argv
    if argv_value(swift_argv, '--warmup_ratio') not in (None, '0', '0.0'):
        sys.exit('train_omni.py: give --warmup_steps or --warmup_ratio, not both')
    if not max_steps or int(max_steps) <= 0:
        sys.exit('train_omni.py: --warmup_steps needs --max_steps (it is passed on as --warmup_ratio, see fix_hf_warmup_steps)')
    ratio = float(steps) / int(max_steps)
    if ratio >= 1:
        sys.exit(f'train_omni.py: --warmup_steps {steps} >= --max_steps {max_steps}')
    out, skip = [], False
    for i, x in enumerate(swift_argv):
        if skip:
            skip = False
            continue
        if x == '--warmup_steps':
            skip = True
            continue
        if x.startswith('--warmup_steps=') or x == '--warmup_ratio' or x.startswith('--warmup_ratio='):
            skip = x == '--warmup_ratio'
            continue
        out.append(x)
    ratio = (int(float(steps)) - 0.5) / int(max_steps)   # ceil(max_steps * ratio) == steps
    print(f'[train_omni] --warmup_steps {steps} -> --warmup_ratio {ratio:.6g} (ms-swift / transformers drop warmup_steps)',
          flush=True)
    return out + ['--warmup_ratio', f'{ratio:.9g}']


def safe_save_defaults(mode, swift_argv):
    """Disk guards, added unless the flag is given explicitly (an explicit flag always wins):
      - megatron: --merge_lora false. Megatron-SWIFT's default (true) writes a full merged model (66 GB) next to
        every LoRA checkpoint, including the final one;
      - both: --save_total_limit 3. Otherwise every checkpoint is kept; one full-fine-tuning Megatron checkpoint
        with optimizer state is ~490 GB."""
    given = {x.split('=')[0] for x in swift_argv if x.startswith('--')}
    extra = []
    if mode == 'megatron' and '--merge_lora' not in given:
        extra += ['--merge_lora', 'false']
    if '--save_total_limit' not in given:
        extra += ['--save_total_limit', '3']
    if extra:
        print(f'[train_omni] disk-safe defaults added: {" ".join(extra)}', flush=True)
    return extra


def parse_audio_layers(spec, n_layers):
    """--audio_lora_layers value -> sorted 0-based layer indices, e.g. 'top8' on 32 layers -> 24..31."""
    s = spec.strip().lower()
    if s in ('', 'none'):
        return []
    if s == 'all':
        return list(range(n_layers))
    for prefix in ('top', 'bottom'):
        if s.startswith(prefix):
            k = int(s[len(prefix):])
            if not 1 <= k <= n_layers:
                raise ValueError(f'--audio_lora_layers {spec}: K must be 1..{n_layers}')
            return list(range(n_layers - k, n_layers)) if prefix == 'top' else list(range(k))
    layers = set()
    for part in s.split(','):
        a, _, b = part.partition('-')
        layers.update(range(int(a), int(b or a) + 1))
    bad = sorted(i for i in layers if not 0 <= i < n_layers)
    if bad or not layers:
        raise ValueError(f'--audio_lora_layers {spec}: layers must be in 0..{n_layers - 1} (got {bad or "none"})')
    return sorted(layers)


def audio_encoder_num_layers(model):
    """encoder_layers of the thinker's audio encoder, from the model's config.json (local dir or HF cache)."""
    path = os.path.join(model, 'config.json')
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, 'config.json')
    with open(path) as f:
        return json.load(f)['thinker_config']['audio_config']['encoder_layers']


def audio_lora_enabled(omni):
    return omni.audio_lora_layers.strip().lower() not in ('', 'none')


def add_audio_lora_targets(omni, swift_argv):
    """Append audio encoder linears to --target_modules for --audio_lora_layers / --audio_lora_modules.

    Names are suffixes such as `audio_tower.layers.31.self_attn.q_proj`. PEFT matches a target when the module name
    ends with it, so the same list works for HF (`thinker.audio_tower...`) and Megatron-SWIFT (`visual.audio_tower...`).
    swift parses --target_modules as one list, and a second --target_modules would replace the first, so the names
    are inserted into the existing list."""
    if not audio_lora_enabled(omni):
        return swift_argv
    if argv_value(swift_argv, '--tuner_type') not in ('lora', None) or argv_value(swift_argv, '--target_regex'):
        sys.exit('train_omni.py: --audio_lora_layers needs --tuner_type lora with --target_modules (no --target_regex)')
    starts = [i for i, x in enumerate(swift_argv) if x == '--target_modules' or x.startswith('--target_modules=')]
    if len(starts) != 1:
        sys.exit('train_omni.py: --audio_lora_layers needs exactly one explicit --target_modules list '
                 '(e.g. $(cat lora_targets_thinker_attn_audio_proj.txt), or linear_qkv linear_proj for Megatron)')
    i = starts[0]
    if swift_argv[i].startswith('--target_modules='):   # split `--target_modules=x` so more names can follow
        swift_argv = swift_argv[:i] + ['--target_modules', swift_argv[i].split('=', 1)[1]] + swift_argv[i + 1:]
    end = i + 1
    while end < len(swift_argv) and not swift_argv[end].startswith('--'):
        end += 1
    if any(t.startswith('all-') for t in swift_argv[i + 1:end]):
        sys.exit('train_omni.py: --audio_lora_layers cannot be combined with all-linear (ms-swift then turns the '
                 'target list into a regex and drops extra names); list the LLM targets explicitly')
    model = argv_value(swift_argv, '--model')
    layers = parse_audio_layers(omni.audio_lora_layers, audio_encoder_num_layers(model))
    extra = [f'audio_tower.layers.{n}.{m}' for n in layers for m in AUDIO_LORA_MODULES[omni.audio_lora_modules]]
    print(f'[train_omni] audio encoder LoRA: layers {layers[0]}..{layers[-1]} ({len(layers)} layers) x '
          f'{omni.audio_lora_modules} -> {len(extra)} extra target modules', flush=True)
    return swift_argv[:end] + extra + swift_argv[end:]


_INIT_ADAPTER = {}   # --adapters: the LoRA this run starts from (recorded in omni_effective.json)


def check_init_adapters(mode, swift_argv):
    """`--adapters <dir>`: continue training an earlier run's LoRA as the new run's starting point. ms-swift sft loads it
    with `from_pretrained(..., is_trainable=True)` (pipelines/train/tuner.py), so only the adapter weights carry over:
    the optimizer, the LR schedule (warmup again) and the data position start fresh; with a new --data_seed the mosaic
    `choose` subsets are drawn anew. The adapter's own adapter_config.json decides rank, alpha, dropout and target
    modules; --lora_rank / --lora_alpha / --target_modules are then ignored by ms-swift, so a mismatch is reported here.
    Not combined with --resume_from_checkpoint (that restores everything, including the data position)."""
    path = argv_value(swift_argv, '--adapters')
    if not path:
        return
    if mode != 'sft':
        sys.exit('train_omni.py megatron: start from a LoRA with --mcore_adapter <dir> --finetune true, not --adapters')
    if argv_value(swift_argv, '--resume_from_checkpoint'):
        sys.exit('train_omni.py: give --adapters (new run from these LoRA weights) or --resume_from_checkpoint '
                 '(exact restart), not both')
    cfg_path = os.path.join(path, 'adapter_config.json')
    if not (os.path.isfile(cfg_path) and os.path.isfile(os.path.join(path, 'adapter_model.safetensors'))):
        sys.exit(f'train_omni.py: --adapters {path}: no adapter_config.json + adapter_model.safetensors')
    with open(cfg_path) as f:
        cfg = json.load(f)
    for flag, key in (('--lora_rank', 'r'), ('--lora_alpha', 'lora_alpha'), ('--lora_dropout', 'lora_dropout')):
        v = argv_value(swift_argv, flag)
        if v is not None and float(v) != float(cfg.get(key, v)):
            print(f'[train_omni] WARNING: {flag} {v} is ignored, the adapter in {path} has {key}={cfg[key]}', flush=True)
    _INIT_ADAPTER['path'] = os.path.abspath(path)
    print(f'[train_omni] init from adapter {path}: r={cfg.get("r")} alpha={cfg.get("lora_alpha")} '
          f'{len(cfg.get("target_modules") or [])} target modules (weights only; fresh optimizer, schedule, data)',
          flush=True)


def fix_peft_target_modules(adapter_dir):
    """adapter_config.json target_modules match more modules than were trained: Megatron-SWIFT writes bare names
    (q_proj, fc1, ...), and swift sft shortens the list (e.g. `28.self_attn.k_proj`, which also matches LLM layer 28).
    PEFT then wraps those extra modules in zero-initialised LoRA when loading. Replace the list with the exact module
    names that have weights in adapter_model.safetensors. Original kept as adapter_config.orig.json."""
    cfg_path = os.path.join(adapter_dir, 'adapter_config.json')
    weights = os.path.join(adapter_dir, 'adapter_model.safetensors')
    if not (os.path.isfile(cfg_path) and os.path.isfile(weights)):
        return
    from safetensors import safe_open
    with safe_open(weights, 'pt') as f:
        names = sorted({k.removeprefix('base_model.model.').rsplit('.lora_', 1)[0] for k in f.keys() if '.lora_' in k})
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not names or sorted(cfg.get('target_modules') or []) == names:
        return
    backup = os.path.join(adapter_dir, 'adapter_config.orig.json')
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            json.dump(cfg, f, indent=2)
    cfg['target_modules'] = names
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'[train_omni] {cfg_path}: target_modules set to the {len(names)} modules in the adapter', flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('sft', 'megatron'):
        sys.exit(__doc__)
    mode, rest = argv[0], argv[1:]
    raw_argv = list(argv)
    if mode == 'megatron':
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')   # as `megatron sft` does
    maybe_relaunch_with_torchrun(argv)
    rest = expand_config(rest)
    omni, swift_argv = parse_omni_args(rest)
    if omni.clean_stale_shm:
        from omni_mds.mosaic_stream import clean_stale_shared_memory
        clean_stale_shared_memory()
    if any(x.split('=')[0] in ('--dataset', '--val_dataset', '--cached_dataset') for x in swift_argv):
        sys.exit('train_omni.py: data comes from --mix; do not pass --dataset / --val_dataset / --cached_dataset')
    from omni_mds.mix import load_mix
    has_val = bool(load_mix(omni.mix)['validation'])

    def arg_is(name, value):
        return f'{name}={value}' in swift_argv or any(
            x == name and y == value for x, y in zip(swift_argv, swift_argv[1:]))

    eval_off = arg_is('--eval_strategy', 'no') or arg_is('--eval_iters', '0')
    swift_argv = fix_hf_warmup_steps(mode, swift_argv)
    swift_argv += safe_save_defaults(mode, swift_argv)
    swift_argv = add_audio_lora_targets(omni, swift_argv)
    check_init_adapters(mode, swift_argv)
    if mode == 'megatron':
        check_megatron_resume_args(swift_argv)
    swift_argv = setup_wandb(mode, omni, swift_argv)
    if omni.wandb:
        _WANDB_EXTRA_CONFIG.update({'omni': vars(omni), 'mix': {split: [spec.__dict__ for spec in specs]
                                                                for split, specs in load_mix(omni.mix).items()}})
    swift_argv += ['--dataset', TRAIN_PLACEHOLDER, '--split_dataset_ratio', '0']
    if has_val and not eval_off:
        swift_argv += ['--val_dataset', VAL_PLACEHOLDER]
    save_run_files(omni, swift_argv, raw_argv)
    apply_gpu_mem_cap()
    if mode == 'sft':
        make_sft_cls(omni)(swift_argv).main()
    else:
        make_megatron_sft_cls(omni)(swift_argv).main()


if __name__ == '__main__':
    main()
