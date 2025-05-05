import dataclasses
import json
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    if k.startswith('prompt_'):
                        to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    else:
                        to_pad = [torch.tensor(ex[k][i], dtype=dtype) for ex in features for i in range(len(ex[k]))]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)

            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

def add_bos_token_if_needed(
    bos_token_id: Optional[int],
    prompt_len_input_ids: int,
    prompt_tokens: Dict[str, List[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: Dict[str, List[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: Dict[str, List[int]],
):
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        
        assert len(chosen_prompt_len_input_ids) == len(chosen_tokens["prompt_input_ids"])
        for i in range(len(chosen_prompt_len_input_ids)):
            if chosen_prompt_len_input_ids[i] == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][i][0]:
                chosen_tokens["prompt_input_ids"][i] = [bos_token_id] + chosen_tokens["prompt_input_ids"][i]
                chosen_tokens["prompt_attention_mask"][i] = [1] + chosen_tokens["prompt_attention_mask"][i]
        
        assert len(rejected_prompt_len_input_ids) == len(rejected_tokens["prompt_input_ids"])
        for i in range(len(rejected_prompt_len_input_ids)):
            if rejected_prompt_len_input_ids[i] == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][i][0]:
                rejected_tokens["prompt_input_ids"][i] = [bos_token_id] + rejected_tokens["prompt_input_ids"][i]
                rejected_tokens["prompt_attention_mask"][i] = [1] + rejected_tokens["prompt_attention_mask"][i]
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int, chosen_tokens: Dict[str, List[int]], rejected_tokens: Dict[str, List[int]]
):
    
    for i in range(len(chosen_tokens["input_ids"])):
        if len(chosen_tokens["input_ids"][i]) == 0 or eos_token_id != chosen_tokens["input_ids"][i][-1]:
            chosen_tokens["input_ids"][i].append(eos_token_id)
            chosen_tokens["attention_mask"][i].append(1)
    
    for i in range(len(rejected_tokens["input_ids"])):
        if len(rejected_tokens["input_ids"][i]) == 0 or eos_token_id != rejected_tokens["input_ids"][i][-1]:
            rejected_tokens["input_ids"][i].append(eos_token_id)
            rejected_tokens["attention_mask"][i].append(1)
    return chosen_tokens, rejected_tokens