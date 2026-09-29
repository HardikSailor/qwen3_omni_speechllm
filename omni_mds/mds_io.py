"""Minimal MDS v2 shard decoding, without the mosaicml-streaming dependency.

Shard layout (MDS v2, all our columns are variable-width str/bytes):
    uint32 num_samples | uint32 offsets[num_samples + 1] | samples
Each sample: uint32 sizes[num_columns] followed by the column bytes, in column order.
Offsets are absolute byte positions in the uncompressed shard.

Column names and encodings come from the shard's entry in index.json. They differ between
shards (34 shards have no `language` column), so always take them per shard.
"""
import json
import os
import struct
from typing import Dict, Iterator, List, Optional

import numpy as np


def load_index(subdir: str) -> List[dict]:
    """Shard entries of one MDS sub-directory (a folder holding index.json)."""
    with open(os.path.join(subdir, 'index.json')) as f:
        return json.load(f)['shards']


def list_subdirs(dataset_dir: str) -> List[str]:
    """Sub-directories holding an index.json (`0/ … 15/`, or the dataset dir itself)."""
    if os.path.isfile(os.path.join(dataset_dir, 'index.json')):
        return [dataset_dir]
    subs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)]
    subs = [d for d in subs if os.path.isfile(os.path.join(d, 'index.json'))]
    return sorted(subs, key=lambda d: (len(os.path.basename(d)), os.path.basename(d)))


def decode_sample(buf: bytes, column_names: List[str], column_encodings: List[str]) -> Dict:
    k = len(column_names)
    sizes = np.frombuffer(buf, np.uint32, k)
    out, pos = {}, 4 * k
    for name, enc, size in zip(column_names, column_encodings, sizes):
        value = buf[pos:pos + int(size)]
        pos += int(size)
        if enc == 'str':
            out[name] = value.decode('utf-8')
        elif enc == 'bytes':
            out[name] = bytes(value)
        else:
            raise ValueError(f'unsupported MDS column encoding {enc!r} for column {name!r}')
    return out


def iter_zstd_shard_head(subdir: str, shard: dict, max_samples: Optional[int] = None) -> Iterator[Dict]:
    """Decode the first `max_samples` samples of a .mds.zstd shard by streaming decompression.

    Only reads as far into the shard as needed, so it is cheap for peeking and tests. It is not
    random access: zstd frames can't be seeked (see MDS_DATA_PIPELINE.md §1).
    """
    import zstandard
    path = os.path.join(subdir, shard['zip_data']['basename'] if shard.get('zip_data') else shard['raw_data']['basename'])
    names, encs = shard['column_names'], shard['column_encodings']
    with open(path, 'rb') as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh) if path.endswith('.zstd') else fh
        n = struct.unpack('<I', _read_exact(reader, 4))[0]
        offsets = np.frombuffer(_read_exact(reader, 4 * (n + 1)), np.uint32)
        pos = 4 + 4 * (n + 1)
        count = n if max_samples is None else min(n, max_samples)
        for i in range(count):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if start > pos:  # should not happen in MDS v2, but stay robust
                _read_exact(reader, start - pos)
            yield decode_sample(_read_exact(reader, end - start), names, encs)
            pos = end


def _read_exact(reader, n: int) -> bytes:
    chunks, remaining = [], n
    while remaining:
        chunk = reader.read(remaining)
        if not chunk:
            raise EOFError(f'unexpected end of shard ({n - remaining}/{n} bytes read)')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)
