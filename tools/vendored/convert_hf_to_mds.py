from streaming import MDSWriter
import numpy as np
import os, sys
from datasets import load_from_disk
from datasets.distributed import split_dataset_by_node
from tqdm import tqdm
from glob import glob
import shutil
import math
from multiprocessing import Pool
from streaming.base.util import merge_index
import torch.distributed as dist
import time
import io
import soundfile as sf
import math
from torch.utils.data import DataLoader

format = "raw"
dataset_dir = "/datasets/datasets_multimodal"
output_dir = f"/datasets/datasets_multimodal_{format}"
splits = ["test"]

def get_text(text):
    return text if text is not None else ""

def get_array(audio, format="array"):
    if format == "opus":
        if audio is None:
            return b''
        else:
            with io.BytesIO() as f:
                f.name="test.ogg"
                sf.write(f, audio["array"], 16000, format="OGG", subtype="OPUS")
                f.seek(0)
                bytes= f.read()
                return bytes
    else:
        return audio["array"] if audio is not None else np.array([0])


def collate_fn(samples):

    output_samples = []

    for sample in samples:
        try:
            output_sample = {
                "context_text": get_text(sample["context"]["text"]),
                "context_audio": get_array(sample["context"]["audio"], format=format),
                "instruction_text": get_text(sample["instruction"]["text"]),
                "instruction_audio": get_array(sample["instruction"]["audio"], format=format),
                "answer_text": get_text(sample["answer"]["text"]),
                "answer_audio": get_array(sample["answer"]["audio"], format=format),
                "task":task,
                "original_dataset_path": dataset_path
            }
            output_samples.append(output_sample)
        except:
            pass
    return output_samples


def convert_to_mds(dataset_path, dataset_output_subpath, rank, world_size, format="array"):

    dataset = load_from_disk(dataset_path)
    orig_len = len(dataset)

    max_world_size = max(min(world_size, math.ceil(orig_len/500)), 1)
    if rank >= max_world_size:
        return None

    audio_type="bytes" if format in ["opus"] else "ndarray"
    # A dictionary of input fields to an Encoder/Decoder type
    columns = {
        "context_text": "str",
        "context_audio": audio_type,
        "instruction_text": "str",
        "instruction_audio": audio_type,
        "answer_text": "str",
        "answer_audio": audio_type,
        "task": "str",
        "original_dataset_path": "str"
    }

    dataset = split_dataset_by_node(dataset, rank, max_world_size)

    dataset_output_subpath
    if len(dataset) == 0:
        return None

    with MDSWriter(
        out=dataset_output_subpath,
        columns=columns,
        compression="zstd",
        size_limit=1024*1024*500,
    ) as out:
        dataloader = DataLoader(dataset, batch_size=1, num_workers=6, collate_fn=collate_fn)
        for samples in tqdm(dataloader):
            for sample in samples:
                try:
                    out.write(sample)
                except:
                    pass

dist.init_process_group("gloo")
torch_rank = dist.get_rank()
torch_world_size = dist.get_world_size()

f = open(os.devnull, "w")
if torch_rank != 0:
    sys.stdout = f
    sys.stderr = f

for split in splits:
    for i, dataset_path in enumerate(sorted(glob(os.path.join(dataset_dir, f"{split}/**/dataset_info.json"), recursive=True))):

        dataset_path = os.path.dirname(dataset_path)
        dataset_base_path = dataset_path.replace(dataset_dir, "")[1:]
        dataset_output_path = os.path.join(output_dir, dataset_base_path)
        if os.path.exists(os.path.join(dataset_output_path, "index.json")):
            continue

        task = dataset_base_path.split("/")[1]
        if torch_rank == 0:
            print(dataset_output_path)
   
        if torch_rank == 0:
            if os.path.exists(dataset_output_path):
                shutil.rmtree(dataset_output_path)
            os.makedirs(dataset_output_path, exist_ok=True)
        dist.barrier()

        convert_to_mds(dataset_path, os.path.join(dataset_output_path, str(torch_rank)), torch_rank, torch_world_size, format=format)

        if torch_rank==0:
            merge_index(dataset_output_path, keep_local=True)
        dist.barrier()
