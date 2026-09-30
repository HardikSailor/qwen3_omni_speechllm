"""Tests for omni_mds.mosaic_stream (PIPELINE_PLAN.md §6, phase 2, reader option C).

CPU only; uses the three small CoVoST2 X->en train sets (~0.1 GB), cached under /tmp/$USER/omni_mds_test.
Needs mosaicml-streaming (container/pydeps, see container/README.md). Run in an interactive job:
    source nscc/env.sh && $RUN python tests/test_mosaic_stream.py
"""
import collections
import gc
import getpass
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from omni_mds.mds_io import iter_zstd_shard_head  # noqa: E402
from omni_mds.mix import DatasetSpec  # noqa: E402
from omni_mds.mosaic_stream import OmniStreamingDataset, StreamConfig, make_dataloader  # noqa: E402
from omni_mds.sea_text import RowConfig, build_row  # noqa: E402

BASE = '/data/projects/13003558/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal/train/ST'
TEST_ROOT = f'/tmp/{getpass.getuser()}/omni_mds_test'
SPECS = [
    DatasetSpec(name='ST_CoVoST2_ta_en_30_ST', path=f'{BASE}/CoVoST2_ta_en_30_ST', task='ST', src_lang='ta', tgt_lang='en'),
    DatasetSpec(name='ST_CoVoST2_id_en_30_ST', path=f'{BASE}/CoVoST2_id_en_30_ST', task='ST', src_lang='id', tgt_lang='en'),
    DatasetSpec(name='ST_CoVoST2_zh-CN_en_30_ST', path=f'{BASE}/CoVoST2_zh-CN_en_30_ST', task='ST', src_lang='zh',
                tgt_lang='en'),
]
SIZES = {'ST_CoVoST2_ta_en_30_ST': 1310, 'ST_CoVoST2_id_en_30_ST': 1243, 'ST_CoVoST2_zh-CN_en_30_ST': 7079}


def with_choose(**choose):
    return [DatasetSpec(**{**s.__dict__, 'choose': choose.get(s.name)}) for s in SPECS]


def make_ds(name, specs=SPECS, **kw):
    """A dataset with its own cache folder (a local dir can't be reused while another dataset holds it)."""
    cache = os.path.join(TEST_ROOT, name)
    shutil.rmtree(cache, ignore_errors=True)
    stream_kw = kw.pop('stream_kw', {})
    cfg = StreamConfig(cache_root=cache, cache_limit=None, shuffle_block_size=None, num_canonical_nodes=1,
                       **stream_kw)
    return OmniStreamingDataset(specs, 'train', RowConfig(), batch_size=kw.pop('batch_size', 4), stream_cfg=cfg,
                                with_meta=True, **kw)


def free(ds):
    del ds
    gc.collect()


# ------------------------------------------------------------------------------------------------ tests

def test_raw_samples_match_direct_decode():
    ds = make_ds('raw', stream_kw={'shuffle': False})
    shard0 = json.load(open(f'{SPECS[0].path}/index.json'))['shards'][0]
    want = list(iter_zstd_shard_head(SPECS[0].path, shard0, 40))
    for sid in range(40):
        got = ds.raw_item(sid)
        assert got == want[sid], sid
        assert ds.stream_of(sid) == 0
    free(ds)


def test_rows_match_build_row():
    ds = make_ds('rows', stream_kw={'shuffle': False})
    for sid in list(range(20)) + [1310 + 5, 1310 + 1243 + 7]:          # samples from all three streams
        spec = SPECS[ds.stream_of(sid)]
        want = build_row(ds.raw_item(sid), spec.meta(), RowConfig(), random.Random(f'42-0-{sid}'), with_meta=True)
        got = ds[sid]
        assert got['_meta'].pop('sample_id') == sid
        assert got == want, sid
        assert got['_meta']['dataset'] == spec.name
    free(ds)


def test_choose_counts_per_epoch():
    choose = {'ST_CoVoST2_ta_en_30_ST': 300, 'ST_CoVoST2_id_en_30_ST': 200, 'ST_CoVoST2_zh-CN_en_30_ST': 500}
    ds = make_ds('choose', specs=with_choose(**choose))
    assert len(ds) == 1000
    epochs = []
    for _ in range(2):
        rows = list(ds)
        counts = collections.Counter(r['_meta']['dataset'] for r in rows)
        assert dict(counts) == choose, counts
        epochs.append({r['_meta']['sample_id'] for r in rows})
        assert len(epochs[-1]) == 1000                                  # without replacement within an epoch
    assert epochs[0] != epochs[1]                                       # 'balanced' re-draws the subset each epoch
    free(ds)


def test_fixed_sampling_for_validation():
    choose = {'ST_CoVoST2_ta_en_30_ST': 64, 'ST_CoVoST2_id_en_30_ST': 64, 'ST_CoVoST2_zh-CN_en_30_ST': 64}
    ds = make_ds('fixed', specs=with_choose(**choose), is_train=False,
                 stream_kw={'shuffle': False, 'sampling_method': 'fixed'})
    a = [r['_meta']['sample_id'] for r in ds]
    b = [r['_meta']['sample_id'] for r in ds]
    assert a == b and len(a) == 192                                     # same subset, same order every epoch
    free(ds)


def _rank_worker(rank, world, out_path):
    """Runs in a subprocess: one of `world` GPUs ranks on a single node, sharing one cache folder
    (the 8-GPU-per-node layout). Two simulated *nodes* can't share one host: mosaic keys its shared memory
    on the host, so a real multi-node check is left for the 2-node run (PIPELINE_PLAN.md phase 4)."""
    # mosaic starts torch.distributed itself when WORLD_SIZE > 1 (torchrun sets these in real runs)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=os.environ.get('TEST_MASTER_PORT', '29577'))
    choose = {'ST_CoVoST2_ta_en_30_ST': 300, 'ST_CoVoST2_id_en_30_ST': 200, 'ST_CoVoST2_zh-CN_en_30_ST': 500}
    cfg = StreamConfig(cache_root=os.path.join(TEST_ROOT, 'ranks'), cache_limit=None, shuffle_block_size=None,
                       num_canonical_nodes=1)
    ds = OmniStreamingDataset(with_choose(**choose), 'train', RowConfig(), batch_size=4, stream_cfg=cfg,
                              with_meta=True)
    ids = [r['_meta']['sample_id'] for r in ds]
    json.dump({'len': len(ds), 'ids': ids}, open(out_path, 'w'))


def test_two_ranks_get_disjoint_equal_partitions():
    outs = [os.path.join(TEST_ROOT, f'rank{r}.json') for r in range(2)]
    shutil.rmtree(os.path.join(TEST_ROOT, 'ranks'), ignore_errors=True)
    os.makedirs(TEST_ROOT, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, __file__, '--rank-worker', str(r), '2', outs[r]]) for r in range(2)]
    assert all(p.wait(timeout=600) == 0 for p in procs)
    res = [json.load(open(o)) for o in outs]
    a, b = set(res[0]['ids']), set(res[1]['ids'])
    assert len(res[0]['ids']) == len(res[1]['ids']) == res[0]['len'] == 500, [len(r['ids']) for r in res]
    assert not (a & b) and len(a | b) == 1000


def test_resume_mid_epoch_gives_same_next_batches():
    choose = {'ST_CoVoST2_ta_en_30_ST': 100, 'ST_CoVoST2_id_en_30_ST': 100, 'ST_CoVoST2_zh-CN_en_30_ST': 100}
    ids = lambda batch: [r['_meta']['sample_id'] for r in batch]  # noqa: E731
    ds = make_ds('resume_a', specs=with_choose(**choose))
    dl = make_dataloader(ds, batch_size=4, collate_fn=ids, num_workers=0, pin_memory=False)
    it = iter(dl)
    for _ in range(5):
        next(it)
    state = dl.state_dict()
    expected = [next(it) for _ in range(3)]
    free(ds)
    ds2 = make_ds('resume_b', specs=with_choose(**choose))
    dl2 = make_dataloader(ds2, batch_size=4, collate_fn=ids, num_workers=0, pin_memory=False)
    dl2.load_state_dict(state)
    got = [b for _, b in zip(range(3), dl2)]
    assert got == expected, (got, expected)
    free(ds2)


def test_dataloader_counts_padding_free_batches():
    """A padding-free collator returns input_ids [1, T]; resume must still count batch_size per batch."""
    import torch
    choose = {'ST_CoVoST2_ta_en_30_ST': 40, 'ST_CoVoST2_id_en_30_ST': 40, 'ST_CoVoST2_zh-CN_en_30_ST': 40}
    ds = make_ds('pf', specs=with_choose(**choose))
    pf = lambda batch: {'input_ids': torch.zeros(1, 10 * len(batch))}  # noqa: E731
    dl = make_dataloader(ds, batch_size=4, collate_fn=pf, num_workers=0, pin_memory=False)
    it = iter(dl)
    for _ in range(3):
        next(it)
    assert dl.num_samples_yielded == 12
    assert dl.state_dict()['sample_in_epoch'] == 12
    free(ds)


def test_unusable_samples_are_replaced_not_dropped():
    choose = {'ST_CoVoST2_ta_en_30_ST': 100, 'ST_CoVoST2_id_en_30_ST': 100, 'ST_CoVoST2_zh-CN_en_30_ST': 100}
    ds = make_ds('replace', specs=with_choose(**choose))
    bad = set(range(0, ds.num_samples, 7))
    orig = ds._make_row

    def failing(sid):
        if sid in bad:
            raise ValueError('simulated bad sample')
        return orig(sid)

    ds._make_row = failing
    rows = list(ds)
    assert len(rows) == 300                                             # count unchanged: ranks stay in step
    assert not any(r['_meta']['sample_id'] in bad for r in rows)
    assert ds.num_replaced > 0
    free(ds)


def throughput_rows_per_s(n=2000, num_workers=0):
    ds = make_ds('speed', batch_size=8)
    dl = make_dataloader(ds, batch_size=8, collate_fn=lambda b: b, num_workers=num_workers, pin_memory=False)
    t0, count = time.time(), 0
    for batch in dl:
        count += len(batch)
        if count >= n:
            break
    dt = time.time() - t0
    free(ds)
    return count / dt


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--rank-worker':
        _rank_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        sys.exit(0)
    import logging
    logging.basicConfig(level=logging.ERROR)
    tests = [test_raw_samples_match_direct_decode, test_rows_match_build_row, test_choose_counts_per_epoch,
             test_fixed_sampling_for_validation, test_two_ranks_get_disjoint_equal_partitions,
             test_resume_mid_epoch_gives_same_next_batches, test_dataloader_counts_padding_free_batches,
             test_unusable_samples_are_replaced_not_dropped]
    failed = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'ok    {fn.__name__} ({time.time() - t0:.1f}s)')
        except Exception as e:
            failed += 1
            import traceback
            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {str(e)[:500]}')
            traceback.print_exc(limit=3)
    print(f'rows/s, first epoch incl. shard copy+unzip, build_row only, 1 process: {throughput_rows_per_s():.0f}')
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    sys.exit(1 if failed else 0)
