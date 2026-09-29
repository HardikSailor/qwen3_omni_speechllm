"""Qwen3-Omni training entry: ms-swift (unmodified) + our MDS data layer (omni_mds, mosaic streaming reader).

    python train_omni.py sft      --mix mixes/st_v0.yaml [omni options] <ordinary `swift sft` arguments>
    python train_omni.py megatron --mix mixes/st_v0.yaml [omni options] <ordinary `megatron sft` arguments>
With NPROC_PER_NODE set it re-launches itself under torchrun, like the swift / megatron CLIs.

Inside the container (PIPELINE_PLAN.md §3.2, §10):
    PYTHONPATH=<this dir>:/scratch/prj0000000234/sailorhb/toolkits/pydeps \
    bash /scratch/prj0000000234/sailorhb/container/run_container.sh \
        python train_omni.py sft --mix mixes/st_v0.yaml --model <path> --tuner_type lora ...

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
    encoder: the layer linears are appended to --target_modules (both trainers).
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
    return p.parse_known_args(argv)


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


# ------------------------------------------------------------------------------------------------ trainer

def make_streaming_trainer_cls(base, fix_adapter_config=False):
    """Subclass of the ms-swift trainer class that reads OmniStreamingDataset with per-rank mosaic dataloaders."""
    from swift.utils import get_logger

    from omni_mds.mosaic_stream import OmniStreamingDataset, make_dataloader
    logger = get_logger()

    class OmniStreamingTrainer(base):
        _omni_train_dl = None
        _omni_eval_dls = None

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
    if mode == 'megatron':
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')   # as `megatron sft` does
    maybe_relaunch_with_torchrun(argv)
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
    swift_argv += safe_save_defaults(mode, swift_argv)
    swift_argv = add_audio_lora_targets(omni, swift_argv)
    if mode == 'megatron':
        check_megatron_resume_args(swift_argv)
    swift_argv += ['--dataset', TRAIN_PLACEHOLDER, '--split_dataset_ratio', '0']
    if has_val and not eval_off:
        swift_argv += ['--val_dataset', VAL_PLACEHOLDER]
    apply_gpu_mem_cap()
    if mode == 'sft':
        make_sft_cls(omni)(swift_argv).main()
    else:
        make_megatron_sft_cls(omni)(swift_argv).main()


if __name__ == '__main__':
    main()
