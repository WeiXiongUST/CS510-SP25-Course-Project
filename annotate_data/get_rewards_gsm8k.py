import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="uf_split0_responses_K8.jsonl",
        metadata={"help": "the location of the dataset name or path for the policy model responses"},
    )
    golden_answer: Optional[str] = field(
        default="",
        metadata={"help": "the location of the dataset name or path with golden answer"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the output file"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ds_dir = script_args.dataset_name_or_path
ds = load_dataset("json", data_files=ds_dir, split="train")
gt = load_dataset(script_args.golden_answer)['train']
gt_dict = {}
for s in gt:
    assert str(s['context_messages']) not in gt_dict
    gt_dict[str(s['context_messages'])] = s['answer']

data_size = len(ds["prompt"])


def get_reward(test_texts, answer):
    short_answer = answer.split('#### ')[-1]
    final_answer = [t.split('Final answer: ')[-1] for t in test_texts]
    rewards = [1 if short_answer in a else 0 for a in final_answer]
    return rewards

def format_gt(answer):
    reasoning = answer.split('####')[0]
    short = answer.split('#### ')[1]
    return reasoning + '\nFinal answer: ' + short

data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        if len(sample["responses"]) < script_args.K:
            continue
        
        answer = gt_dict[str(sample['prompt'])]
        rewards = get_reward(sample['responses'], answer)
                
        data.append({"prompt": sample["prompt"], "responses": sample["responses"] + [format_gt(answer)], "rewards": rewards + [1]})

all_rewards = [sample["rewards"] for sample in data]
sum_scores = np.mean(np.sum(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)
max_scores = np.max(np.sum(all_rewards, axis=1))

print(
    "Collect {} data from {} inputs. mean score {} sum score: {} max score: {}".format(
        len(data), data_size, mean_scores, sum_scores, max_scores
    )
)

if len(data) < data_size:
    print(
        "Some of the prompts are with responses < {}. This can happen because the prompt is too long and is ignored by VLLM".format(
            script_args.K
        )
    )

with open(script_args.output_dir, "w", encoding="utf8") as f:
    for i in range(len(data)):
        json.dump(data[i], f, ensure_ascii=False)
        f.write('\n')
