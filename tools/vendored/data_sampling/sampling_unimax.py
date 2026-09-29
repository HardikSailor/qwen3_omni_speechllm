'''
Return dataset config with sample size computed by UniMax (https://arxiv.org/pdf/2304.09151)
with the following modifications:
- sample size can be adjusted at task and/or dataset level
- option to exclude tasks i.e. these tasks and their datasets will be assigned their original sample size
'''

import os
import argparse
import yaml, json

def unimax_sampling(size_dict, budget, N, exclude_task):
    '''
    UniMax sampling adapted from https://arxiv.org/pdf/2304.09151
    Acts at both task and dataset level
    '''
    # initialize
    sample_dict = {k: 0 for k in size_dict}
    # unimax
    sort_ascending = {k: v for k, v in sorted(size_dict.items(), key=lambda item: item[1])}
    i = 0
    for k in sort_ascending.keys():
        if k in exclude_task:
            sample_k = size_dict[k]
        else:
            budget_per_item = budget / (len(sort_ascending) - i)
            size_k = size_dict[k]
            if budget_per_item > size_k * N:
                sample_k = int(size_k * N)
            else:
                sample_k = int(budget_per_item)
        sample_dict[k] = sample_k
        budget -= sample_k
        i += 1
    return sample_dict

def unimax_task_dataset(task_dataset_dict, budget, task_N, dataset_N, exclude_task=None):
    # sampling by task
    task_dict = {task: sum(task_i_dict.values()) for task, task_i_dict in task_dataset_dict.items()}
    budget = sum(task_dict.values()) if budget is None else budget
    task_sample_dict = unimax_sampling(task_dict, budget, task_N, exclude_task)

    # for each task, sampling by dataset
    task_dataset_sample_dict = {task: {} for task in task_dataset_dict}
    for task, task_budget in task_sample_dict.items():
        # update dataset sample size by task budget
        task_scaling = task_budget / sum(task_dataset_dict[task].values())
        task_i_dict = {k: v * task_scaling for k, v in task_dataset_dict[task].items()}

        # sampling by dataset
        if task in exclude_task:
            task_dataset_sample_dict[task] = task_i_dict
        else:
            dataset_sample_dict = unimax_sampling(task_i_dict, task_budget, dataset_N, [])
        task_dataset_sample_dict[task] = dataset_sample_dict
    return task_dataset_sample_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--count_save_dir", type=str,
        default=None,
        help="Path to directory where count results are saved.")
    parser.add_argument("--count_json_save_name", type=str,
        default="mds_datasets_v2_sample_size",
        help="Name of file where count results are saved.")
    parser.add_argument("--dataset_config_root", type=str,
        default=None,
        help="Path to dataset config directory.")
    parser.add_argument("--dataset_config_yaml", type=str,
        default="stage2-partial_v2",
        help="Path to a dataset config yaml file.")
    parser.add_argument("--exclude_task", nargs='*', default=["AQA", "SQA", "ASQA"],
        help="Datasets in these tasks are assigned their original sample size.")
    parser.add_argument("--budget", type=int, default=None,
        help="Sample size per epoch. If None, use the total number of samples.")
    parser.add_argument("--task_multiplier", type=float, default=1.5,
        help="Maximum sample size multiplier for a task.")
    parser.add_argument("--dataset_multiplier", type=float, default=1.0,
        help="Maximum sample size multiplier for a dataset. Compounded with task_multiplier.")
    parser.add_argument("--save_suffix", type=str, default="unimax")
    args = parser.parse_args()
    print(args)

    args.count_save_dir = os.path.dirname(os.path.abspath(__file__)) if args.count_save_dir is None else args.count_save_dir

    # load count file
    count_path = os.path.join(args.count_save_dir, args.count_json_save_name + '.json')
    with open(count_path, 'r') as file:
        count_dict = json.load(file)

    # load dataset config
    if args.dataset_config_root is None:
        repo_parent = args.count_save_dir.split("multimodal_trainer")[0]
        args.dataset_config_root = os.path.join(repo_parent, "multimodal_trainer/config/dataset")
    dataset_config_path = os.path.join(args.dataset_config_root, args.dataset_config_yaml + ".yaml")
    with open(dataset_config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    # get count for training datasets in dataset config
    train_count_dict = {}
    for df in config["train"]:
        task = df["task"]
        data_name = df["path"].split("/")[-1]
        if task not in train_count_dict:
            train_count_dict[task] = {}
        if data_name not in train_count_dict[task]:
            train_count_dict[task][data_name] = 0
        train_count_dict[task][data_name] = count_dict[task][data_name]["num_samp"]

    # sample
    train_sample_dict = unimax_task_dataset(train_count_dict,
        budget=args.budget, task_N=args.task_multiplier, dataset_N=args.dataset_multiplier, exclude_task=args.exclude_task)

    # update config
    for df in config["train"]:
        task = df["task"]
        data_name = df["path"].split("/")[-1]
        df['num_samp'] = train_count_dict[task][data_name]
        df["num_choose"] = train_sample_dict[task][data_name]

    # save
    save_path = os.path.join(args.dataset_config_root, f"{args.dataset_config_yaml}_{args.save_suffix}-B{args.budget}-T{args.task_multiplier}-D{args.dataset_multiplier}.yaml")

    class YamlDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(YamlDumper, self).increase_indent(flow, False)

        def write_line_break(self, data=None):
            if len(self.indents) <= 2:  # Only add an extra line break at the top level
                super().write_line_break()
            super().write_line_break()

    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, Dumper=YamlDumper)

    print(f"Sampling results are saved to {save_path}.")