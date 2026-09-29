from streaming import MDSWriter,StreamingDataset
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import shutil
import math
from multiprocessing import Pool
from streaming.base.util import merge_index
import sys
import soundfile as sf
import tempfile
import torch.distributed as dist
from torch.utils.data import DataLoader
from streaming.base.util import clean_stale_shared_memory 
from streaming.base.util import merge_index
from functools import partial
import io


dataset_dir = "/datasets"
output_dir = "/home/datasets/mds_datasets_ogg"
cache_dir = "/home/datasets/mds_cache"

def convert_text(text):
        return text

def convert_audio(audio):
    if len(audio) == 1:
        return b''
    else:
        with io.BytesIO() as f:
            f.name="test.ogg"
            sf.write(f, audio, 16000)
            f.seek(0)
            bytes= f.read()
        return bytes


def data_collator(samples, task):
    ret_samples = []
    for sample in samples:
        try:
            ret_sample = {
            "context_text": convert_text(sample["context_text"]),
            "context_audio": convert_audio(sample["context_audio"]),
            "instruction_text": convert_text(sample["instruction_text"]),
            "instruction_audio": convert_audio(sample["instruction_audio"]),
            "answer_text": convert_text(sample["answer_text"]),
            "answer_audio": convert_audio(sample["answer_audio"]),
            "task":task,
            "original_dataset_path": dataset_path
            }
            ret_samples.append(ret_sample)
        except:
            print(sample)
            pass


    return ret_samples



def convert_to_mds(dataset_output_path, dataloader) -> None:

    # A dictionary of input fields to an Encoder/Decoder type
    columns = {
        "context_text": "str",
        "context_audio": "bytes",
        "instruction_text": "str",
        "instruction_audio": "bytes",
        "answer_text": "str",
        "answer_audio": "bytes",
        "task": "str",
        "original_dataset_path": "str"
    }

    with MDSWriter(
        out=os.path.join(dataset_output_path, str(dist.get_rank())),
        columns=columns,
        compression="zstd",
        size_limit=1024*1024*100,
    ) as out:
        for batch in tqdm(dataloader):
            for sample in batch:
                out.write(sample)


dist.init_process_group()


for dataset_path in glob(os.path.join(dataset_dir, f"**/index.json"), recursive=True):
    if dataset_path.split("/")[-2].isdigit():
        continue
    if "invalidated" in dataset_path:
        continue
    dataset_path = os.path.dirname(dataset_path)
    dataset_base_path = dataset_path.replace(dataset_dir, "")[1:]
    dataset_output_path = os.path.join(output_dir, dataset_base_path)
    task = dataset_path.split("/")[-2]


    if dist.get_rank() == 0:
        print(dataset_path)
        print(dataset_base_path)
        print(task)
        if os.path.exists(dataset_output_path):
            shutil.rmtree(dataset_output_path)
        os.makedirs(dataset_output_path, exist_ok=True)
    dist.barrier()

    num_workers = 2
    batch_size = 8
    dataset = StreamingDataset(remote=dataset_path, local=os.path.join(cache_dir, dataset_base_path), predownload=10*batch_size, batch_size=batch_size, shuffle=False, shuffle_block_size=250000, num_canonical_nodes=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(data_collator, task=task), drop_last=False, num_workers=0)

    # Create PyTorch DataLoader
    convert_to_mds(dataset_output_path, dataloader)
    dist.barrier()

    if dist.get_rank() == 0:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        merge_index(dataset_output_path, keep_local=True)
    clean_stale_shared_memory()
    dist.barrier()
