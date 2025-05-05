import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser

"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    base_path: Optional[str] = field(
        default="/home/xiongwei/gshf/data/gen_data",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    num_datasets: Optional[int] = field(
        default=8,
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


all_dirs = [script_args.base_path + "_data" + str(i+1) + ".jsonl" for i in range(script_args.num_datasets)]

data_dict = {}
for file in all_dirs:
    tmp_data = read_jsonl(file)
    for d in tmp_data:
        prompt_str = str(d['prompt'])
        if prompt_str not in data_dict:
            data_dict[prompt_str] = d
        else:
            data_dict[prompt_str]['responses'] = data_dict[prompt_str]['responses'] + d['responses']

# save
with open(script_args.output_dir, 'w') as fout:
    for p in data_dict:
        fout.write(json.dumps(data_dict[p]) + '\n')
