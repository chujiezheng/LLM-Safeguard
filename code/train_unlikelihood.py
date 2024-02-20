import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from typing import Union, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from safetensors import safe_open
import gc
import random
from typing import List
from matplotlib import pyplot as plt
from safetensors.torch import save_file
from copy import deepcopy
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
from utils import PCA_DIM


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")

BATCH_SIZE = 50
NUM_EPOCHES = 5
MAX_RESPONSE_LENGTH = 100


def embed_soft_prompt(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_messages: List[List[Dict[str, str]]],
    soft_prompt: torch.Tensor
):
    if soft_prompt.device != model.device:
        raise ValueError("soft_prompt must be on the same device as model")
    if soft_prompt.dtype != model.dtype:
        raise ValueError("soft_prompt must be of the same dtype as model")

    if soft_prompt.dim() != 2:
        raise ValueError("soft_prompt must be a 2D tensor")
    if any(len(messages) != 2 for messages in all_messages):
        raise ValueError("all_messages must be a list of single-message lists")
    n_prompt_tokens = soft_prompt.size(0)

    # As system message appears first, we replace the first n_prompt_tokens eos tokens with soft_prompt
    messages_with_eos_placeholder = [[{'role': 'system', 'content': toker.eos_token * n_prompt_tokens}] + e for e in all_messages]
    input_ids_w = [toker.apply_chat_template(e, add_generation_prompt=True, tokenize=True) for e in messages_with_eos_placeholder]
    input_ids_wo = [toker.apply_chat_template(e[:-1], add_generation_prompt=False, tokenize=True) for e in messages_with_eos_placeholder]

    input_ids = []
    lm_labels = []
    for ipt_ids_w, ipt_ids_wo in zip(input_ids_w, input_ids_wo):
        ipt_ids_resp = ipt_ids_w[len(ipt_ids_wo):]
        ipt_ids = ipt_ids_wo + ipt_ids_resp[:MAX_RESPONSE_LENGTH+1]
        lm_labs = [-100] * len(ipt_ids_wo) + ipt_ids_resp[:MAX_RESPONSE_LENGTH+1]
        ipt_ids = ipt_ids[:-1]
        lm_labs = lm_labs[1:]
        input_ids.append(ipt_ids)
        lm_labels.append(lm_labs)

    input_lengths = [len(e) for e in input_ids]
    max_input_length = max(input_lengths)
    input_ids = [e + [toker.eos_token_id] * (max_input_length - len(e)) for e in input_ids]
    lm_labels = [e + [-100] * (max_input_length - len(e)) for e in lm_labels]

    placeholder_start_index = input_ids[0].index(toker.eos_token_id) # all input_ids have the same placeholder_start_index
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    lm_labels = torch.tensor(lm_labels, dtype=torch.long).to(model.device)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds[:, placeholder_start_index:placeholder_start_index+n_prompt_tokens] = soft_prompt.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
    return inputs_embeds, lm_labels


def get_shuffled_messages_and_labels(all_messages: List[List[Dict[str, str]]], labels: torch.Tensor, seed=42):
    rng = random.Random(seed)
    assert len(all_messages) == len(labels)
    indices = list(range(len(all_messages)))
    for epoch_idx in range(NUM_EPOCHES):
        rng.shuffle(indices)
        for idx in range(len(all_messages)//BATCH_SIZE):
            yield epoch_idx, [all_messages[indices[idx*BATCH_SIZE + i]] for i in range(BATCH_SIZE)], labels[indices[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]]


def get_data(model_name, dataset, prompt_length):
    data = []

    # for harmful queries
    eval_results_file = f'./eval_results/sampling/{model_name}_with_{prompt_length}_{dataset}_all.csv'
    df = pd.read_csv(eval_results_file) # prompt, output, refusal_score
    for index, row in df.iterrows():
        prompt = row['prompt']
        output = row['output']
        refusal_score = row['refusal_score']
        data.append({'prompt': prompt, 'output': output, 'label': int(refusal_score == 0)})

    eval_results_file = f'./eval_results/sampling/{model_name}_{dataset}_all.csv'
    df = pd.read_csv(eval_results_file) # prompt, output, refusal_score
    for index, row in df.iterrows():
        prompt = row['prompt']
        output = row['output']
        refusal_score = row['refusal_score']
        # only use negative samples
        if refusal_score == 0:
            continue
        data.append({'prompt': prompt, 'output': output, 'label': int(refusal_score == 0)})

    # for harmless queries
    eval_results_file = f'./eval_results_harmless/sampling/{model_name}_with_{prompt_length}_{dataset}_all.csv'
    df = pd.read_csv(eval_results_file) # prompt, output, refusal_score
    for index, row in df.iterrows():
        prompt = row['prompt']
        output = row['output']
        refusal_score = row['refusal_score']
        data.append({'prompt': prompt, 'output': output, 'label': int(refusal_score == 1)})

    eval_results_file = f'./eval_results_harmless/sampling/{model_name}_{dataset}_all.csv'
    df = pd.read_csv(eval_results_file) # prompt, output, refusal_score
    for index, row in df.iterrows():
        prompt = row['prompt']
        output = row['output']
        refusal_score = row['refusal_score']
        # only use negative samples
        if refusal_score == 1:
            continue
        data.append({'prompt': prompt, 'output': output, 'label': int(refusal_score == 1)})

    return data


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short'], required=True)
    parser.add_argument("--output_path", type=str, default='./trained_prompts_unlikelihood')
    args = parser.parse_args()

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # prepare model
    model_name = args.pretrained_model_path.split('/')[-1]

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else None,
    )
    device = model.device
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    os.makedirs(f'{args.output_path}/{model_name}', exist_ok=True)

    # prepare toker
    toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast='Orca-2-' not in model_name)

    if 'Llama-2-' in model_name and '-chat' in model_name:
        generation_config_file = './generation_configs/llama-2-chat.json'
    elif 'CodeLlama-' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/llama-2-chat.json'
    elif 'Orca-2-' in model_name:
        generation_config_file = './generation_configs/orca-2.json'
    elif 'Mistral-' in model_name and '-Instruct' in model_name:
        generation_config_file = './generation_configs/mistral-instruct.json'
    elif 'vicuna-' in model_name:
        generation_config_file = './generation_configs/vicuna.json'
    elif 'openchat-' in model_name:
        generation_config_file = './generation_configs/openchat.json'
    else:
        raise ValueError(f"Unsupported or untuned model: {model_name}")
    generation_config = json.load(open(generation_config_file))
    chat_template_file = generation_config['chat_template']
    chat_template = open(chat_template_file).read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    toker.chat_template = chat_template

    # prepare soft prompt
    if args.prompt_length == 'default':
        init_ids = toker(DEFAULT_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'short':
        init_ids = toker(SHORT_SYSTEM_PROMPT).input_ids[1:]
    elif args.prompt_length == 'mistral':
        init_ids = toker(MISTRAL_SYSTEM_PROMPT).input_ids[1:]
    init_embeds = model.get_input_embeddings().weight.data[init_ids].detach()
    soft_prompt = nn.Parameter(init_embeds, requires_grad=True).to(model.device)

    # prepare data
    data = get_data(model_name, 'custom', args.prompt_length)

    all_messages = [[{'role': 'user', 'content': e['prompt'].strip()}, {'role': 'assistant', 'content': e['output'].strip()}] for e in data]
    cls_labels = [e['label'] for e in data]
    cls_labels = torch.tensor(cls_labels, dtype=torch.long, device=device)

    step = 0
    optimizer = torch.optim.AdamW([soft_prompt], lr=1e-3)
    seed = 42
    for epoch_idx, batch_messages, batch_cls_labels in get_shuffled_messages_and_labels(all_messages, cls_labels, seed=seed):
        optimizer.zero_grad()
        inputs_embeds, lm_labels = embed_soft_prompt(model, toker, batch_messages, soft_prompt)
        batch_cls_labels = batch_cls_labels.unsqueeze(1)

        lm_logits = model(inputs_embeds=inputs_embeds).logits
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1), reduction='none')
        loss = loss.view(lm_labels.size(0), lm_labels.size(1)) * batch_cls_labels
        label_size = (lm_labels.ne(-100) * batch_cls_labels).sum(1).type_as(loss)
        lm_loss = loss.sum() / torch.clamp(label_size.sum(), min=1e-5)

        neg_logits = torch.clamp(1. - F.softmax(lm_logits, dim=-1), min=1e-5).log()
        neg_loss = F.cross_entropy(neg_logits.view(-1, neg_logits.size(-1)), lm_labels.view(-1), reduction='none')
        neg_loss = neg_loss.view(lm_labels.size(0), lm_labels.size(1)) * (1. - batch_cls_labels)
        neg_label_size = (lm_labels.ne(-100) * (1. - batch_cls_labels)).sum(1).type_as(loss)
        neg_lm_loss = neg_loss.sum() / torch.clamp(neg_label_size.sum(), min=1e-5)

        if model_name in ['Llama-2-7b-chat-hf', 'CodeLlama-7b-Instruct-hf']:
            total_loss = lm_loss + 5 * neg_lm_loss
        else:
            total_loss = lm_loss + neg_lm_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(soft_prompt, 1.0)
        optimizer.step()
        step += 1

        if step % 10 == 0:
            logging.info(f'Step {step}, lm_loss {lm_loss.cpu().item()}, neg_lm_loss {neg_lm_loss.cpu().item()}')

    soft_prompt = soft_prompt.detach()
    save_file({'soft_prompt': soft_prompt}, f'{args.output_path}/{model_name}/length.{args.prompt_length}.safetensors')

    logging.info(f"Training finished")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
