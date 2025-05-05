import os
import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    base_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    num_datasets: Optional[int] = field(
        default=1,
        metadata={"help": "the location of the output file"},
    )

def read_jsonl(file):
    data = []
    with open(file) as f:
        readin = f.readlines()
        for line in readin:
            tmp = json.loads(line)
            data.append(tmp)
    return data

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
num_datasets = script_args.num_datasets

data_dir = os.path.dirname(script_args.base_path)
current_iteration = script_args.base_path.split('/')[-1][4]
tmp_iteration = int(current_iteration)
if num_datasets == 0:
    num_datasets = int(current_iteration)

data_dict = {}
cnt = 0
while tmp_iteration > 0 and cnt != num_datasets:
    
    tmp_data = read_jsonl(os.path.join(data_dir, f'Test{tmp_iteration}_Iter{tmp_iteration}_reward.json'))
    for d in tmp_data:
        prompt_str = str(d['prompt'])
        if prompt_str not in data_dict:
            data_dict[prompt_str] = d
        else:
            data_dict[prompt_str]['responses'] = data_dict[prompt_str]['responses'] + d['responses']
            data_dict[prompt_str]['rewards'] = data_dict[prompt_str]['rewards'] + d['rewards']
    
    cnt += 1
    tmp_iteration -= 1

# save
with open(os.path.join(data_dir, f'Test{current_iteration}_Iter{current_iteration}_train_reward.json'), 'w') as fout:
    for p in data_dict:
        fout.write(json.dumps(data_dict[p]) + '\n')
