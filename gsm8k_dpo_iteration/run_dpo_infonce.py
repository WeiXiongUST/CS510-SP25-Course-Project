import os
from dataclasses import dataclass, field
from typing import Optional

import random
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from alignment import H4ArgumentParser
from trl import (
    DPOConfig,
    # DPOTrainer,
    ModelConfig,
)

from trl.commands.cli_utils import TrlParser
from dpo_infonce import DPOTrainer
from itertools import cycle

def minimal_duplication_sampling(A, b):
    # Shuffle the list for randomness
    shuffled_A = A.copy()
    random.shuffle(shuffled_A)
    
    # Use cycle to repeat the list elements once all are used
    cycling_elements = cycle(shuffled_A)
    
    # Select b elements
    result = [next(cycling_elements) for _ in range(b)]
    return result

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default="/export/home/hanze/project/vllm-gen/uf_split0_offline_reward.json",  # "/export/home/data/gemma_it_2b_3w_k8_with_pairrm_rewards.json",
        metadata={"help": "the location of the evalset name or path"},
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_min", metadata={"help": "the choose type"})
    positive_N: Optional[int] = field(default=1, metadata={"help": "the number of positive responses sampled"})
    negative_N: Optional[int] = field(default=1, metadata={"help": "the number of negative responses sampled"})

    # # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})


def prepare_data(
    data_dir: str = "/home/xiongwei/data/helpful/rm/rm1003.jsonl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
    positive_N=1,
    negative_N=1,
) -> Dataset:
    """Prepare the dataset for DPO training by rejection sampling.
    We implement different strategies to select pairs, including
    max_min: best v.s. worst
    max_random: best v.s. random from the remaining;
    max_max: best v.s. second best
    max_min_p: best v.s. worst but we additionally add a length penalty in the reward value
    """
    ds = load_dataset("json", data_files=data_dir, split="train")
    print(ds)

    pos = []
    neg = []
    prompts = []

    margin = []
    
    assert len(ds[0]["rewards"]) >= positive_N + negative_N
    for sample in ds:
        P = tokenizer.apply_chat_template(sample["prompt"], tokenize = False, add_generation_prompt= True)
        if choose_type == "random":
            idx0 = list(range(positive_N))
            idx1 = list(range(positive_N, positive_N + negative_N))
        elif choose_type == "max_min":
            # sorted_index = np.array(sample["rewards"]).argsort()
            # idx0 = sorted_index[-positive_N:]
            # idx1 = sorted_index[:negative_N]
            
            pos_idx = []
            neg_idx = []
            for i, score in enumerate(sample["rewards"]):
                if score == 1:
                    pos_idx.append(i)
                elif score == 0:
                    neg_idx.append(i)
                else:
                    raise ValueError('Reward should be 0 or 1.')
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            idx0 = minimal_duplication_sampling(pos_idx, positive_N)
            idx1 = minimal_duplication_sampling(neg_idx, negative_N)

        elif choose_type == "other":
                        
            pos_idx = []
            for i, score in enumerate(sample["rewards"]):
                if score == 1:
                    pos_idx.append(i)
                elif score == 0:
                    pass
                else:
                    raise ValueError('Reward should be 0 or 1.')
            
            neg_responses = []
            while len(neg_responses) < negative_N:
                random_index = random.randint(0, len(ds)-1)
                random_example = ds[random_index]
                random_id = random.randint(0, len(random_example["rewards"])-1)
                neg_responses.append(random_example["responses"][random_id])

            if len(pos_idx) == 0 or len(neg_responses) == 0:
                continue
            idx0 = minimal_duplication_sampling(pos_idx, positive_N)
            # idx1 = minimal_duplication_sampling(neg_idx, negative_N)

        else:
            raise NotImplementedError

        if choose_type == "other":
            prompts.append(P)
            pos.append([sample["responses"][idd] + eot_token for idd in idx0])
            neg.append([neg_response + eot_token for neg_response in neg_responses])
            margin.append((sample["rewards"][idx0[-1]] - 0) * margin_scale)
        else:
            prompts.append(P)
            pos.append([sample["responses"][idd] + eot_token for idd in idx0])
            neg.append([sample["responses"][idd] + eot_token for idd in idx1])
            margin.append((sample["rewards"][idx0[-1]] - sample["rewards"][idx1[0]]) * margin_scale)

    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":

    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = model_config.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        model_ref.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        model_ref.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))

    # 2. Load the Stack-exchange paired dataset
    train_dataset = prepare_data(
        data_dir=script_args.train_dir,
        margin_scale=script_args.margin_scale,
        sanity_check=script_args.sanity_check,
        choose_type=script_args.choose_type,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
        positive_N=script_args.positive_N,
        negative_N=script_args.negative_N,
    )

    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        data_dir=script_args.eval_dir,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
        positive_N=script_args.positive_N,
        negative_N=script_args.negative_N,
    )

    # 4. initialize training arguments:
    print(training_args)

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta=training_args.beta,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )
    print("begin to train")

    # from IPython import embed
    # embed()
    # exit()

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    # 7. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
