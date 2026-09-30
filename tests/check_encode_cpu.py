"""CPU check of the full data path without the model: mosaic -> build_row -> ms-swift Qwen3-Omni template encode
-> padding-free collator. Checks audio token counts (~13/s), prompt masking in labels and batch shapes.
    source nscc/env.sh && $RUN python tests/check_encode_cpu.py      # in an interactive job
"""
import functools, getpass, glob, io, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf
from swift import get_processor, get_template
from omni_mds.mix import load_mix
from omni_mds.mosaic_stream import OmniStreamingDataset, StreamConfig, make_dataloader
from omni_mds.sea_text import RowConfig
from train_omni import encode_row

MODEL = glob.glob('/scratch/users/astar/ares/sailorhb/container/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/*')[0]
proc = get_processor(MODEL, model_type='qwen3_omni_moe')
template = get_template(proc, max_length=2048, truncation_strategy='raise', padding_free=True)
template.set_mode('train')
tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
audio_pad = tok.convert_tokens_to_ids('<|audio_pad|>')

mix = load_mix(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mixes', 'st_v0.yaml'))
specs = [type(s)(**{**s.__dict__, 'choose': 8}) for s in mix['validation']]   # 8 per CoVoST2 test set
cache = f'/tmp/{getpass.getuser()}/omni_encode_check'; shutil.rmtree(cache, ignore_errors=True)
ds = OmniStreamingDataset(specs, 'validation', RowConfig(), batch_size=4, is_train=False,
                          stream_cfg=StreamConfig.for_validation(cache_root=cache, cache_limit=None, num_canonical_nodes=1),
                          with_meta=True)

# 1) per-sample: audio tokens vs duration, label masking
t0 = time.time(); n = 0; ratios = []
for sid_row in ds:
    row = dict(sid_row); meta = row['_meta']
    enc = encode_row(template, row)
    ids, labels = enc['input_ids'], enc['labels']
    n_audio = sum(1 for t in ids if t == audio_pad)
    dur = sf.info(io.BytesIO(row['audios'][0])).duration
    ratios.append(n_audio / dur)
    n_lab = sum(1 for l in labels if l != -100)
    target = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    assert row['messages'][1]['content'] in target, (target, row['messages'][1]['content'])
    assert all(l == -100 for t, l in zip(ids, labels) if t == audio_pad)          # audio never in the loss
    if n < 2 or meta['src_lang'] == 'ta' and n % 8 == 0:
        print(f"{meta['dataset']:28s} {dur:5.1f}s  audio_tokens={n_audio:4d} ({n_audio/dur:4.1f}/s) len={enc['length']:4d} "
              f"label_tokens={n_lab:3d} target={target[:60]!r}")
    n += 1
print(f'encoded {n} rows in {time.time()-t0:.1f}s ({n/(time.time()-t0):.1f} rows/s, 1 CPU process incl. feature extraction); '
      f'audio tokens/s: min {min(ratios):.1f} mean {sum(ratios)/len(ratios):.1f} max {max(ratios):.1f}')
print('keys:', sorted(enc.keys()))

# 2) through the dataset transform + dataloader + padding-free collator
ds2 = OmniStreamingDataset(specs, 'validation', RowConfig(), batch_size=4, is_train=False,
                           stream_cfg=StreamConfig.for_validation(cache_root=cache + '2', cache_limit=None,
                                                                   num_canonical_nodes=1),
                           transform=functools.partial(encode_row, template))
dl = make_dataloader(ds2, batch_size=4, collate_fn=template.data_collator, num_workers=2, pin_memory=False)
for i, batch in enumerate(dl):
    if i == 0:
        print('batch:', {k: (tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__) for k, v in batch.items()})
print(f'batches: {i+1}, dataloader counted {dl.num_samples_yielded} samples (expected {len(ds2)})')
assert dl.num_samples_yielded == len(ds2)
shutil.rmtree(cache, ignore_errors=True); shutil.rmtree(cache + '2', ignore_errors=True)
print('OK')
