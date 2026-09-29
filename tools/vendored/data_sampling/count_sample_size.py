'''
Count sample size for each dataset.
'''

import argparse
import os, sys
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,
        default='/mnt/home/zoux/mds_datasets_v2/datasets_multimodal/train',
        help="Path to dataset directory.")
    parser.add_argument("--count_save_dir", type=str,
        default=None,
        help="Path to directory to save count results.")
    parser.add_argument("--count_json_save_name", type=str,
        default="mds_datasets_v2_sample_size",
        help="Name of file to save count results.")
    args = parser.parse_args()

    args.count_save_dir = os.path.dirname(os.path.abspath(__file__)) if args.count_save_dir is None else args.count_save_dir

    # load count file
    count_dict = {}
    count_path = os.path.join(args.count_save_dir, args.count_json_save_name + '.json')

    # list all task folders
    task_name_list = [name for name in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, name))]

    # for each task, process each dataset
    for task_name in task_name_list:
        task_folder = os.path.join(args.dataset_dir, task_name)
        dataset_name_list = [name for name in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, name))]

        for dataset_name in dataset_name_list:
            print(f'Counting {dataset_name}...')

            if task_name not in count_dict:
                count_dict[task_name] = {dataset_name: {}}
            else:
                count_dict[task_name][dataset_name] = {}

            index_path = os.path.join(task_folder, dataset_name, "index.json")
            index = json.load(open(index_path, "r"))
            index_shards = index["shards"]
            num_samp = sum([shard["samples"] for shard in index_shards])

            # update results
            count_dict[task_name][dataset_name] = {
                "num_samp": num_samp,
            }

            # save as json
            with open(count_path, 'w') as file:
                json.dump(count_dict, file, indent=4)

    print(f'Count results are saved to {count_path}.')