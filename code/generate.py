import os
import json
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from typing import Union
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from utils import DEFAULT_SYSTEM_PROMPT, SHORT_SYSTEM_PROMPT, MISTRAL_SYSTEM_PROMPT
import gc
import random
from multiprocessing.pool import ThreadPool
from safetensors import safe_open
from functools import partial


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")

BAD_WORDS = [
    '\nHello', '\nHi', # for Orca-2
    ' ON', "I' " # for vicuna
]


def prepend_sys_prompt(sentence, args):
    messages = [{'role': 'user', 'content': sentence.strip()}]
    if args.use_soft_prompt:
        messages = [{'role': 'system', 'content': ''.join([f'<soft_prompt_{i}>' for i in range(args.soft_prompt.size(0))])}] + messages
    elif args.use_default_prompt:
        messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}] + messages
    elif args.use_short_prompt:
        messages = [{'role': 'system', 'content': SHORT_SYSTEM_PROMPT}] + messages
    elif args.use_mistral_prompt:
        messages = [{'role': 'system', 'content': MISTRAL_SYSTEM_PROMPT}] + messages
    return messages


def process_soft_prompt_as_word_embedding(
    model: PreTrainedModel,
    toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    soft_prompt: torch.nn.Parameter
) -> nn.Module:
    # We embed soft prompt into input word embedding and safe it
    # When loaded later, simply call model.set_input_embeddings()
    config = model.config
    padding_idx = config.pad_token_id

    old_toker_size = len(toker)
    toker.add_tokens([f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))], special_tokens=True)
    new_toker_size = len(toker)

    old_input_embeddings = model.get_input_embeddings()
    embedding_dim = old_input_embeddings.embedding_dim
    old_num_embeddings = old_input_embeddings.num_embeddings
    new_num_embeddings = max(new_toker_size, old_num_embeddings)

    new_input_embeddings = nn.Embedding(new_num_embeddings, embedding_dim, padding_idx)
    new_input_embeddings.weight.data[:old_toker_size] = old_input_embeddings.weight.data[:old_toker_size]
    new_input_embeddings.weight.data[old_toker_size:new_toker_size] = soft_prompt.data.to('cpu')
    return toker, new_input_embeddings


def generate(inputs, model, toker, max_new_tokens, n_samples, temp, top_p, stop_token_ids, stop_str):
    qdx, (seed, query, messages) = inputs
    if seed is None:
        set_seed(qdx)
    else:
        set_seed(seed)

    input_text = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(
        toker.convert_tokens_to_ids(toker.tokenize(input_text)),
        dtype=torch.long,
    ).unsqueeze(0).to(model.device)

    # directly tokenizing words would produce an extra space, so remove it
    bad_words_ids = [toker.convert_tokens_to_ids(toker.tokenize(t)[1:]) for t in BAD_WORDS]

    generations = model.generate(
        input_ids,
        attention_mask=input_ids.new_ones(input_ids.size(), dtype=model.dtype),
        min_new_tokens=10,
        max_new_tokens=max_new_tokens,
        do_sample=True if temp > 0 else False,
        temperature=temp if temp > 0 else 1.0,
        top_p=top_p,
        top_k=50, # consistent with hf default value
        num_return_sequences=n_samples,
        eos_token_id=stop_token_ids,
        pad_token_id=toker.eos_token_id,
        return_dict_in_generate=True,
        bad_words_ids=bad_words_ids,
    )
    generations = generations.sequences[..., input_ids.size(1):] # [batch_size, seq_len]

    generations = generations.tolist()
    generated_texts = []
    for generation in generations:
        gen_tokens = []
        for token in generation:
            if token in stop_token_ids or token == toker.eos_token_id:
                break
            gen_tokens.append(token)

        text = toker.decode(gen_tokens, skip_special_tokens=True)
        if stop_str is not None:
            text = text.split(stop_str)[0]
        text = text.strip()

        while (
            not any(text.endswith(e) for e in ['.', '?', '!']) or
            (len(text) > 1 and text[-1] == '.' and text[-2].isdigit())
        ):
            if len(text.split('\n')) > 1:
                text = '\n'.join(text.split('\n')[:-1])
                text = text.strip()
                continue
            break

        generated_texts.append(text)

    return qdx, query, input_text, generated_texts


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--use_sampling", action="store_true")

    parser.add_argument("--use_soft_prompt", action="store_true")
    parser.add_argument("--prompt_length", type=str, choices=['default', 'mistral', 'short'])
    parser.add_argument("--system_prompt_type", type=str, choices=['all', 'default', 'mistral', 'short'])
    parser.add_argument("--do_data_ablation", action="store_true")
    parser.add_argument("--do_unlikelihood", action="store_true")
    parser.add_argument("--ablate_norm", action="store_true")
    parser.add_argument("--ablate_refu", action="store_true")
    parser.add_argument("--ablate_harm", action="store_true")

    parser.add_argument("--use_default_prompt", action='store_true')
    parser.add_argument("--use_short_prompt", action='store_true')
    parser.add_argument("--use_mistral_prompt", action='store_true')

    parser.add_argument("--use_malicious", action="store_true")
    parser.add_argument("--use_advbench", action="store_true")
    parser.add_argument("--use_alpaca", action="store_true")
    parser.add_argument("--use_gcg", action="store_true")
    parser.add_argument("--use_harmless", action="store_true")
    parser.add_argument("--use_testset", action="store_true")

    parser.add_argument("--output_path", type=str, default='./outputs')
    args = parser.parse_args()

    if sum([args.use_soft_prompt,
            args.use_default_prompt,
            args.use_short_prompt, args.use_mistral_prompt]) > 1:
        raise ValueError("Only one of --use_soft_prompt, --use_default_prompt, --use_short/--use_mistral_prompt can be set to True")
    if not args.use_sampling and args.n_samples > 1:
        raise ValueError("n_samples must be 1 in greedy decoding")
    if sum([args.use_malicious, args.use_advbench, args.use_alpaca]) > 1:
        raise ValueError("Only one of --use_malicious/--use_advbench/--use_alpaca can be set to True")
    if any([args.use_malicious, args.use_advbench, args.use_alpaca]) and args.use_harmless:
        raise ValueError("Only one of --use_malicious/--use_advbench/--use_alpaca and --use_harmless can be set to True")
    if any([args.use_malicious, args.use_advbench, args.use_alpaca]) and args.use_testset:
        raise ValueError("Only one of --use_malicious/--use_advbench/--use_alpaca and --use_testset can be set to True")
    if args.use_testset and not args.use_harmless:
        raise ValueError("--use_testset must be used with --use_harmless")

    # prepare toker
    model_name = args.model_name = args.pretrained_model_path.split('/')[-1]
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

    stop_token_ids = generation_config['stop_token_ids']
    if stop_token_ids is None:
        stop_token_ids = [toker.eos_token_id]
    stop_str = generation_config['stop_str']

    # prepare data
    fname = model_name
    if args.use_soft_prompt:
        if args.do_unlikelihood:
            fname += f"_with_soft_unlikelihood_{args.prompt_length}"
        else:
            fname += f"_with_soft_{args.system_prompt_type}_{args.prompt_length}"
        if args.do_data_ablation:
            fname += '_ablation'
            fname += '_unlikelihood'
        elif args.ablate_norm:
            fname += "_nonorm"
        elif args.ablate_refu:
            fname += "_norefu"
        elif args.ablate_harm:
            fname += "_noharm"
    elif args.use_default_prompt:
        fname += "_with_default"
    elif args.use_short_prompt:
        fname += "_with_short"
    elif args.use_mistral_prompt:
        fname += "_with_mistral"

    if args.use_harmless:
        data_path = './data_harmless'
        args.output_path += "_harmless"
    elif args.use_alpaca:
        data_path = './data_alpaca'
        args.output_path += "_alpaca"
    else:
        data_path = './data'

    if args.use_advbench:
        fname += "_advbench"
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
    elif args.use_malicious:
        fname += "_malicious"
        with open(f"{data_path}/MaliciousInstruct.txt") as f:
            lines = f.readlines()
    elif args.use_gcg:
        fname += "_gcg"
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
        templates = json.load(open(f'{data_path}/gcg.json'))
        for i in range(len(lines)):
            template = templates[str(i)]['final_suffix']
            lines[i] = lines[i] + template
    elif args.use_testset:
        fname += "_testset"
        with open(f"{data_path}/testset.txt") as f:
            lines = f.readlines()
    elif args.use_alpaca:
        fname += "_alpaca"
        with open(f"{data_path}/alpaca_eval.json") as f:
            lines = [e['instruction'] for e in json.load(f)[:100]]
    else:
        fname += "_custom"
        with open(f"{data_path}/custom.txt") as f:
            lines = f.readlines()
    os.makedirs(f"{args.output_path}/{fname}", exist_ok=True)

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    if os.path.exists(f"{args.output_path}/{fname}/output_sampling.csv" if args.use_sampling else f"{args.output_path}/{fname}/output_greedy.csv"):
        logging.info(f"File {args.output_path}/{fname}/output_sampling.csv exists, skipping")
        return

    # prepare model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
                and not ((('Orca-2-' in model_name and args.use_soft_prompt)
                          or ('vicuna-' in model_name and not args.use_soft_prompt)
                          ) and args.use_testset)
                else torch.float32,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported()
                and not (('Orca-2-' in model_name or 'vicuna-' in model_name) and args.use_testset)
                else None,
    )

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging_cuda_memory_usage()

    if args.use_soft_prompt:
        if args.do_data_ablation:
            soft_prompt_file = f'./trained_prompts_ablation/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors'
        elif args.do_unlikelihood:
            soft_prompt_file = f'./trained_prompts_unlikelihood/{model_name}/length.{args.prompt_length}.safetensors'
        elif args.ablate_norm:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_nonorm.safetensors'
        elif args.ablate_refu:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_norefu.safetensors'
        elif args.ablate_harm:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}_noharm.safetensors'
        else:
            soft_prompt_file = f'./trained_prompts/{model_name}/type.{args.system_prompt_type}_length.{args.prompt_length}.safetensors'
        with safe_open(soft_prompt_file, framework='pt') as f:
            soft_prompt = f.get_tensor('soft_prompt')
        args.soft_prompt = soft_prompt
        toker, new_input_embeddings = process_soft_prompt_as_word_embedding(model, toker, soft_prompt)
        model.set_input_embeddings(new_input_embeddings.to(device=model.device, dtype=model.dtype))

    # prepend sys prompt
    all_queries = [l.strip() for l in lines]
    all_messages = [prepend_sys_prompt(l, args) for l in all_queries]
    if args.use_autodan or args.use_gcg:
        with open(f"{data_path}/advbench.txt") as f:
            lines = f.readlines()[:100]
        all_queries = [l.strip() for l in lines]

    logging.info(f"Running")
    prompts = []
    inputs = []
    outputs = []
    model.eval()

    if args.use_harmless:
        max_new_tokens = 200
    elif args.use_alpaca:
        max_new_tokens = 1000
    else:
        max_new_tokens = 300

    if args.use_sampling:
        generate_fn = partial(
            generate, model=model, toker=toker,
            max_new_tokens=max_new_tokens,
            n_samples=args.n_samples, temp=1, top_p=0.9,
            stop_token_ids=stop_token_ids, stop_str=stop_str,
        )
    else:
        generate_fn = partial(
            generate, model=model, toker=toker,
            max_new_tokens=max_new_tokens,
            n_samples=1, temp=0, top_p=0,
            stop_token_ids=stop_token_ids, stop_str=stop_str,
        )

    pool = ThreadPool(1)

    seeds = [None] * len(all_queries) # by default, we use qdx
    pbar = tqdm(total=len(all_queries), dynamic_ncols=True)
    for res in pool.imap(generate_fn, enumerate(zip(seeds, all_queries, all_messages)), chunksize=1):
        qdx, query, input_text, generated_texts = res
        if qdx < 5:
            logging.info(f"\nQuery: {query}")
            logging.info(f"\nInput: {input_text}")
            logging.info(f"\nOutput: {generated_texts[0]}\n")
        inputs.extend([input_text] * args.n_samples)
        outputs.extend(generated_texts)
        prompts.extend([query] * args.n_samples)
        pbar.update(1)

    results = pd.DataFrame()
    results["prompt"] = prompts
    results["input"] = inputs
    results["output"] = outputs
    if args.use_sampling:
        results.to_csv(f"{args.output_path}/{fname}/output_sampling.csv")
    else:
        results.to_csv(f"{args.output_path}/{fname}/output_greedy.csv")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
