"""Reader option C (chosen 2026-09-28): mosaic StreamingDataset over the MDS mix, one partition per rank.

Each dataset in the mix YAML becomes one mosaic `Stream`:
  remote = the dataset folder on /scratch (read only; its top-level index.json covers the 0/ … 15/ sub-folders),
  local  = a node-local cache folder; mosaic copies each shard there and unzips it on first use.
`choose` / `repeat` are passed to the Stream, so per-epoch subsampling behaves as in the old trainer
(`multimodal_trainer/modules/datasets/datasets.py` `init_train_stream` / `init_train_streaming_dataset`).

`OmniStreamingDataset.get_item` turns each raw MDS sample into an ms-swift row with `sea_text.build_row`
(and optionally encodes it with `transform`, e.g. the ms-swift template's `encode`). A sample that has no audio
or fails `transform` is **replaced** by a nearby good sample rather than dropped: mosaic gives every rank the
same number of samples, and dropping would leave ranks with different batch counts (DDP would hang).

Partitioning follows mosaic's env vars (RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE), i.e. the global rank.
That matches our layouts where every GPU is its own data-parallel rank (DDP; Megatron EP with TP=PP=CP=1).
With TP/CP > 1 these must be set to the data-parallel rank/size instead (PIPELINE_PLAN.md §8.4).
"""
import getpass
import logging
import os
import random
import socket
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from streaming import Stream, StreamingDataLoader, StreamingDataset

from .mix import DatasetSpec
from .sea_text import RowConfig, build_row

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = os.environ.get('OMNI_MDS_CACHE') or f'/tmp/{getpass.getuser()}/omni_mds_cache'


@dataclass
class StreamConfig:
    cache_root: str = DEFAULT_CACHE_ROOT     # node-local disk (/tmp = /raid NVMe here); a host folder is added
    cache_limit: Optional[str] = '600gb'     # per node, as in the old trainer; None = unlimited
    predownload: Optional[int] = None        # samples to fetch ahead per worker; None = mosaic default (8 x batch)
    shuffle: bool = True
    shuffle_algo: str = 'py1e'
    shuffle_seed: int = 9176
    shuffle_block_size: Optional[int] = 1_500_000   # old trainer value
    num_canonical_nodes: Optional[int] = None       # None = number of nodes at launch (old: world_size // 8)
    sampling_method: str = 'balanced'               # 'fixed' = the same `choose` subset every epoch (validation)
    max_replace_tries: int = 16                     # neighbours tried when a sample is unusable

    @classmethod
    def for_validation(cls, **kw) -> 'StreamConfig':
        return cls(**{'shuffle': False, 'sampling_method': 'fixed', **kw})


def local_dir(cache_root: str, split: str, name: str) -> str:
    # Host name in the path: a cache folder must never be shared between nodes (mosaic coordinates
    # the ranks of one node through shared memory keyed on this path).
    return os.path.join(cache_root, socket.gethostname(), split, name)


def build_streams(specs: List[DatasetSpec], split: str, cache_root: str) -> List[Stream]:
    return [Stream(remote=spec.path, local=local_dir(cache_root, split, spec.name), choose=spec.choose,
                   repeat=spec.repeat, keep_zip=False) for spec in specs]


def default_num_canonical_nodes() -> int:
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    return max(1, world // max(1, local_world))


class OmniStreamingDataset(StreamingDataset):
    """Yields ms-swift rows (or encoded rows, if `transform` is given) from an MDS mix."""

    def __init__(self, specs: List[DatasetSpec], split: str, row_cfg: RowConfig, *, batch_size: int,
                 is_train: bool = True, seed: int = 42, stream_cfg: Optional[StreamConfig] = None,
                 transform: Optional[Callable[[Dict], Optional[Dict]]] = None, augment: bool = False,
                 with_meta: bool = False):
        cfg = stream_cfg or (StreamConfig() if is_train else StreamConfig.for_validation())
        super().__init__(
            streams=build_streams(specs, split, cfg.cache_root),
            batch_size=batch_size,
            shuffle=cfg.shuffle,
            shuffle_algo=cfg.shuffle_algo,
            shuffle_seed=cfg.shuffle_seed,
            shuffle_block_size=cfg.shuffle_block_size,
            num_canonical_nodes=cfg.num_canonical_nodes or default_num_canonical_nodes(),
            cache_limit=cfg.cache_limit,
            predownload=cfg.predownload,
            sampling_method=cfg.sampling_method,
        )
        self.specs, self.split, self.row_cfg, self.is_train = specs, split, row_cfg, is_train
        self.seed, self.transform, self.with_meta = seed, transform, with_meta
        self.augment, self.max_replace_tries = augment, cfg.max_replace_tries
        self._metas = [spec.meta() for spec in specs]
        self._augmenter = None  # created lazily, one per dataloader worker process
        self.num_replaced = 0

    # -- helpers ------------------------------------------------------------------------------------
    def stream_of(self, sample_id: int) -> int:
        shard_id, _ = self.spanner[sample_id]
        return int(self.stream_per_shard[shard_id])

    def current_epoch(self) -> int:
        return max(0, getattr(self, 'next_epoch', 1) - 1)

    def sample_rng(self, sample_id: int) -> random.Random:
        """Per-sample RNG: prompts/dropout are reproducible for (seed, epoch, sample) and re-drawn each epoch."""
        return random.Random(f'{self.seed}-{self.current_epoch()}-{sample_id}')

    def raw_item(self, sample_id: int) -> Dict:
        return super().get_item(sample_id)

    def _make_row(self, sample_id: int) -> Optional[Dict]:
        augment = None
        if self.augment and self.is_train:
            if self._augmenter is None:
                from .audio.augment import AudioAugmenter
                self._augmenter = AudioAugmenter()
            augment = self._augmenter
        raw = self.raw_item(sample_id)
        row = build_row(raw, self._metas[self.stream_of(sample_id)], self.row_cfg, self.sample_rng(sample_id),
                        is_train=self.is_train, with_meta=self.with_meta, augment=augment)
        if row is None or self.transform is None:
            return row
        return self.transform(row)

    # -- mosaic hook --------------------------------------------------------------------------------
    def get_item(self, sample_id: int, retry: int = 7) -> Dict:
        for k in range(self.max_replace_tries):
            sid = (sample_id + k) % self.num_samples   # neighbours are usually in the same (already local) shard
            try:
                row = self._make_row(sid)
            except Exception as e:  # bad audio, encode error, too long for max_length, …
                logger.warning('omni_mds: sample %d (%s) unusable: %s: %s', sid,
                               self.specs[self.stream_of(sid)].name, type(e).__name__, e)
                row = None
            if row is not None:
                if k:
                    self.num_replaced += 1
                if _SAMPLE_ID_LOG and self.is_train:
                    self._log_sample_id(sid)
                if self.with_meta and '_meta' in row:
                    row['_meta']['sample_id'] = int(sid)  # plain int: _meta is written to JSONL by eval
                return row
        raise RuntimeError(f'omni_mds: {self.max_replace_tries} consecutive unusable samples from {sample_id}')


class OmniStreamingDataLoader(StreamingDataLoader):
    """StreamingDataLoader that counts samples as `batch_size` per batch.

    mosaic infers the count from the collated batch (`len(first dict value)`, else `len(batch[0])`). With
    ms-swift's padding-free collator `input_ids` is [1, total_tokens], so it would count 1 per batch and
    resume from the wrong position. Every batch here holds exactly `batch_size` samples: `drop_last=True`,
    and unusable samples are replaced, not dropped.
    """

    def _get_batch_size(self, batch) -> int:
        return self.batch_size


_SAMPLE_ID_LOG = os.environ.get('OMNI_LOG_SAMPLE_IDS')  # debug: folder for per-rank/worker sample-id files


def _log_sample_id_impl(self, sid):
    """Append a used training sample id to <OMNI_LOG_SAMPLE_IDS>/rank<R>_pid<P>.txt (tests: disjoint partitions)."""
    if getattr(self, '_sid_file', None) is None:
        os.makedirs(_SAMPLE_ID_LOG, exist_ok=True)
        self._sid_file = open(os.path.join(_SAMPLE_ID_LOG, f'rank{os.environ.get("RANK", "0")}_pid{os.getpid()}.txt'),
                              'a', buffering=1)
    self._sid_file.write(f'{int(sid)}\n')


OmniStreamingDataset._log_sample_id = _log_sample_id_impl


def make_dataloader(dataset: OmniStreamingDataset, *, batch_size: int, collate_fn: Optional[Callable] = None,
                    num_workers: int = 4, pin_memory: bool = True, prefetch_factor: Optional[int] = 2,
                    persistent_workers: bool = True) -> OmniStreamingDataLoader:
    """Per-rank dataloader with state_dict()/load_state_dict() for mid-epoch resume."""
    if dataset.batch_size != batch_size:
        raise ValueError(f'dataset batch_size {dataset.batch_size} != dataloader batch_size {batch_size}; '
                         'mosaic partitions by the dataset batch size')
    kwargs = {}
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    return OmniStreamingDataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                                   pin_memory=pin_memory, drop_last=True, **kwargs)


def clean_stale_shared_memory() -> None:
    """Remove mosaic shared memory left by crashed jobs, before building any dataset (mosaic does the work on
    local rank 0 and makes the other local ranks wait).

    Caution: it removes *every* mosaic shared-memory segment this user owns on the node, including those of
    another of our jobs still running there. Only call it when the job has the node to itself."""
    from streaming.base.util import clean_stale_shared_memory as _clean
    _clean()
