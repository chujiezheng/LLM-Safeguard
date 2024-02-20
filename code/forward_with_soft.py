import os
import json
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
from safetensors.torch import save_file
import gc
import random
from safetensors import safe_open
from matplotlib import pyplot as plt
from generate import process_soft_prompt_as_word_embedding


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def prepend_sys_prompt(sentence, prompt_length):
    messages = [{'role': 'user', 'content': sentence.strip()}]
    messages = [{'role': 'system', 'content': ''.join([f'<soft_prompt_{i}>' for i in range(prompt_length)])}] + messages
    return messages


def forward(model, toker, messages):
    input_text = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(
        toker.convert_tokens_to_ids(toker.tokenize(input_text)),
        dtype=torch.long,
    ).unsqueeze(0).to(model.device)

    outputs = model(
        input_ids,
        attention_mask=input_ids.new_ones(input_ids.size(), dtype=model.dtype),
        return_dict=True,
        output_hidden_states=True,
    )
    hidden_states = [e[0].detach().half().cpu() for e in outputs.hidden_states[1:]]

    return hidden_states


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short'], default='default')
    parser.add_argument("--system_prompt_type", type=str, choices=['all', 'default', 'mistral', 'short'], default='all')

    parser.add_argument("--use_malicious", action="store_true")
    parser.add_argument("--use_advbench", action="store_true")
    parser.add_argument("--use_harmless", action="store_true")
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--output_path", type=str, default='./hidden_states_with_soft')
    args = parser.parse_args()

    if args.use_malicious and args.use_advbench:
        raise ValueError("Only one of --use_malicious and --use_advbench can be set to True")
    if (args.use_malicious or args.use_advbench) and args.use_harmless:
        raise ValueError("Only one of --use_malicious/--use_advbench and --use_harmless can be set to True")

    # prepare model
    model_name = args.model_name = args.pretrained_model_path.split('/')[-1]

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else None,
    )

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

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

    with safe_open(f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors', framework='pt') as f:
        soft_prompt = f.get_tensor('soft_prompt')
    toker, new_input_embeddings = process_soft_prompt_as_word_embedding(model, toker, soft_prompt)
    model.set_input_embeddings(new_input_embeddings.to(device=model.device, dtype=model.dtype))

    # prepare data
    if args.use_harmless or args.use_testset:
        data_path = './data_harmless'
        args.output_path += "_harmless"
    else:
        data_path = './data'

    if args.use_malicious:
        dataset = "malicious"
        with open(f"{data_path}/MaliciousInstruct.txt") as f:
            lines = f.readlines()
    elif args.use_advbench:
        dataset = "advbench"
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
    elif args.use_testset:
        dataset = "testset"
        with open(f"{data_path}/testset.txt") as f:
            lines = f.readlines()
    else:
        dataset = "custom"
        with open(f"{data_path}/custom.txt") as f:
            lines = f.readlines()
    os.makedirs(f"{args.output_path}", exist_ok=True)

    # prepend sys prompt
    all_queries = [e.strip() for e in lines]
    n_queries = len(all_queries)

    all_messages = [prepend_sys_prompt(l, soft_prompt.size(0)) for l in all_queries]

    logging.info(f"Running")
    tensors = {}
    for idx, messages in tqdm(enumerate(all_messages),
                              total=len(all_messages), dynamic_ncols=True):
        hidden_states = forward(model, toker, messages)
        for i, hs in enumerate(hidden_states):
            tensors[f'sample.{idx}_layer.{i}'] = hs
    save_file(tensors, f'{args.output_path}/{model_name}_{dataset}.safetensors')

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
