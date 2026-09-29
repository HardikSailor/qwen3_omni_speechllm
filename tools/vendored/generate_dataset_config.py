from argparse import ArgumentParser
from glob import glob
import os
import yaml
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--audio_length", type=int, required=False, default=30)
    return parser.parse_args()

possible_length = [30, 60, 120, 300]

if __name__ == "__main__":
    args = parse_args()
    root_path = args.dataset_path
    output_name = args.output_name
    task = args.task

    train_path = os.path.join(root_path, "train")
    val_path = os.path.join(root_path, "test")

    train_datasets = glob(os.path.join(train_path, "**/*"), recursive=False)
    val_datasets = glob(os.path.join(val_path, "**/*"), recursive=False)
    print(train_datasets)

    if task == "ASR":
        train_datasets = [t for t in train_datasets if "ASR" in t]
        val_datasets = [t for t in val_datasets if "ASR" in t]
    elif task == "AC":
        train_datasets = [t for t in train_datasets if "AC" in t]
        val_datasets = [t for t in val_datasets if "AC" in t]
    elif task == "all":
        pass
    else:
        raise ValueError(f"Invalid task: {task}")

    print(train_datasets)
    print(val_datasets)

    train_dict = []
    # val_dict = []

    for dataset in train_datasets:
        name = dataset.split("/")[-1]
        is_skip = False
        for length in possible_length:
            if f"_{length}_" in name and length > args.audio_length:
                print(f'{name} is longer than {args.audio_length} seconds, skipping')
                is_skip = True
                break
        if is_skip:
            continue
        task = dataset.split("/")[-2]
        path = os.path.relpath(dataset, root_path)
        train_dict.append({
            "name": name,
            "task": task,
            "path": path,
            "proportion": 1.0
        })
    
    for train_ds in train_dict:
        train_ds["weight"] = 1 / len(train_dict)

    val_dict = []
    for dataset in val_datasets:
        name = dataset.split("/")[-1]
        is_skip = False
        for length in possible_length:
            if f"_{length}_" in name and length > args.audio_length:
                print(f'{name} is longer than {args.audio_length} seconds, skipping')
                is_skip = True
                break
        if is_skip:
            continue
        task = dataset.split("/")[-2]
        path = os.path.relpath(dataset, root_path)
        val_dict.append({
            "name": name,
            "task": task,
            "path": path,
            "proportion": 1.0
        })

    for val_ds in val_dict:
        val_ds["weight"] = 1 / len(val_dict)

    config_dict = {
        "train": train_dict,
        "validation": val_dict,
    }
    with open(output_name, "w") as f:
        yaml.dump(config_dict, f, indent=4)
