import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from utils import PCA_DIM


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def calculate_boundary(xlim, ylim, weight, bias):
    if np.abs(weight[0]) > np.abs(weight[1]):
        xlim_by_ylim_0 = (-bias - weight[1] * ylim[0]) / weight[0]
        xlim_by_ylim_1 = (-bias - weight[1] * ylim[1]) / weight[0]
        return [(xlim_by_ylim_0, ylim[0]), (xlim_by_ylim_1, ylim[1])]
    else:
        ylim_by_xlim_0 = (-bias - weight[0] * xlim[0]) / weight[1]
        ylim_by_xlim_1 = (-bias - weight[0] * xlim[1]) / weight[1]
        return [(xlim[0], ylim_by_xlim_0), (xlim[1], ylim_by_xlim_1)]


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_paths", type=str, nargs='+', required=True)
    parser.add_argument("--system_prompt_type", type=str, choices=['all'], required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # prepare data
    fname = f'{args.system_prompt_type}_harmfulness_boundary'
    fname += "_custom"
    dataset = 'custom'
    with open(f"./data/custom.txt") as f:
        lines = f.readlines()
    with open(f"./data_harmless/custom.txt") as f:
        lines_harmless = f.readlines()
    os.makedirs(args.output_path, exist_ok=True)

    #colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        'harmless': 'tab:blue',
        'harmful': 'tab:red',
        'harmless + default': 'tab:cyan',
        'harmful + default': 'tab:pink',
        'harmless + mistral': 'tab:olive',
        'harmful + mistral': 'tab:purple',
        'harmless + short': 'tab:brown',
        'harmful + short': 'tab:orange',
    }

    all_queries = [e.strip() for e in lines if e.strip()]
    n_queries = len(all_queries)

    all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]
    n_queries_harmless = len(all_queries_harmless)

    ncols = 4
    if len(args.pretrained_model_paths) % ncols != 0:
        raise ValueError(f"len(args.pretrained_model_paths) % ncols != 0")
    nrows = len(args.pretrained_model_paths) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 4.5 * nrows))

    for mdx, pretrained_model_path in enumerate(args.pretrained_model_paths):
        logging_cuda_memory_usage()
        torch.cuda.empty_cache()
        gc.collect()

        logging.info(pretrained_model_path)

        # prepare model
        model_name = pretrained_model_path.split('/')[-1]
        config = AutoConfig.from_pretrained(pretrained_model_path)
        num_layers = config.num_hidden_layers


        # w/o
        logging.info(f"Running w/o")
        hidden_states = safe_open(f'hidden_states_harmless/{model_name}_{dataset}.safetensors',
                                  framework='pt', device=0)
        all_hidden_states_harmless = []
        for idx, query in enumerate(all_queries):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_harmless.append(tmp_hidden_states)

        hidden_states = safe_open(f'hidden_states/{model_name}_{dataset}.safetensors',
                                  framework='pt', device=0)
        all_hidden_states = []
        for idx, query in enumerate(all_queries):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states.append(tmp_hidden_states)


        all_hidden_states = torch.stack(all_hidden_states)
        all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)

        indices, other_indices = get_following_indices(
            model_name, config=args.config, use_harmless=False)
        indices_harmless, other_indices_harmless = get_following_indices(
            model_name, config=args.config, use_harmless=True)


        # default
        logging.info(f"Running default")
        hidden_states_with_default = safe_open(f'hidden_states_harmless/{model_name}_with_default_{dataset}.safetensors',
                                                        framework='pt', device=0)
        all_hidden_states_with_default_harmless = []
        for idx, query_harmless in enumerate(all_queries_harmless):
            tmp_hidden_states = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_default_harmless.append(tmp_hidden_states)

        hidden_states_with_default = safe_open(f'hidden_states/{model_name}_with_default_{dataset}.safetensors',
                                                framework='pt', device=0)
        all_hidden_states_with_default = []
        for idx, query in enumerate(all_queries):
            tmp_hidden_states = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_default.append(tmp_hidden_states)


        all_hidden_states_with_default = torch.stack(all_hidden_states_with_default)
        all_hidden_states_with_default_harmless = torch.stack(all_hidden_states_with_default_harmless)

        indices_with_default, other_indices_with_default = get_following_indices(
            model_name, config=args.config, use_default_prompt=True, use_harmless=False)
        indices_with_default_harmless, other_indices_with_default_harmless = get_following_indices(
            model_name, config=args.config, use_default_prompt=True, use_harmless=True)


        # mistral
        logging.info(f"Running mistral")
        hidden_states_with_mistral = safe_open(f'hidden_states_harmless/{model_name}_with_mistral_{dataset}.safetensors',
                                                        framework='pt', device=0)
        all_hidden_states_with_mistral_harmless = []
        for idx, query_harmless in enumerate(all_queries_harmless):
            tmp_hidden_states = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_mistral_harmless.append(tmp_hidden_states)

        hidden_states_with_mistral = safe_open(f'hidden_states/{model_name}_with_mistral_{dataset}.safetensors',
                                                framework='pt', device=0)
        all_hidden_states_with_mistral = []
        for idx, query in enumerate(all_queries):
            tmp_hidden_states = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_mistral.append(tmp_hidden_states)


        all_hidden_states_with_mistral = torch.stack(all_hidden_states_with_mistral)
        all_hidden_states_with_mistral_harmless = torch.stack(all_hidden_states_with_mistral_harmless)

        indices_with_mistral, other_indices_with_mistral = get_following_indices(
            model_name, config=args.config, use_mistral_prompt=True, use_harmless=False)
        indices_with_mistral_harmless, other_indices_with_mistral_harmless = get_following_indices(
            model_name, config=args.config, use_mistral_prompt=True, use_harmless=True)


        # short
        logging.info(f"Running short")
        hidden_states_with_short = safe_open(f'hidden_states_harmless/{model_name}_with_short_{dataset}.safetensors',
                                                        framework='pt', device=0)
        all_hidden_states_with_short_harmless = []
        for idx, query_harmless in enumerate(all_queries_harmless):
            tmp_hidden_states = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_short_harmless.append(tmp_hidden_states)

        hidden_states_with_short = safe_open(f'hidden_states/{model_name}_with_short_{dataset}.safetensors',
                                                framework='pt', device=0)
        all_hidden_states_with_short = []
        for idx, query in enumerate(all_queries):
            tmp_hidden_states = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
            all_hidden_states_with_short.append(tmp_hidden_states)


        all_hidden_states_with_short = torch.stack(all_hidden_states_with_short)
        all_hidden_states_with_short_harmless = torch.stack(all_hidden_states_with_short_harmless)

        indices_with_short, other_indices_with_short = get_following_indices(
            model_name, config=args.config, use_short_prompt=True, use_harmless=False)
        indices_with_short_harmless, other_indices_with_short_harmless = get_following_indices(
            model_name, config=args.config, use_short_prompt=True, use_harmless=True)


        hidden_states = torch.cat([
            all_hidden_states_harmless,
            all_hidden_states_with_default_harmless,
            all_hidden_states_with_mistral_harmless,
            all_hidden_states_with_short_harmless,
            all_hidden_states,
            all_hidden_states_with_default,
            all_hidden_states_with_mistral,
            all_hidden_states_with_short,
        ], dim=0).float()

        pca = PCA(PCA_DIM, random_state=42)
        pca.fit(hidden_states.cpu().numpy())
        mean = torch.tensor(pca.mean_, device='cuda', dtype=torch.float)
        V = torch.tensor(pca.components_.T, device='cuda', dtype=torch.float)
        logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, sum: {np.sum(pca.explained_variance_ratio_)}")


        ax = fig.add_subplot(nrows, ncols, mdx + 1)
        ax.set_title(model_name)
        ax.set_aspect(1)

        # harmless
        points = torch.matmul(all_hidden_states_harmless - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_harmless, 0], points[other_indices_harmless, 1],
                    marker='o', alpha=0.38,
                    color=colors['harmless'])
        ax.scatter(points[indices_harmless, 0], points[indices_harmless, 1],
                    marker='o', alpha=0.39,
                    color=colors['harmless'], label='harmless')

        points = torch.matmul(all_hidden_states_with_default_harmless - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_with_default_harmless, 0], points[other_indices_with_default_harmless, 1],
                    marker='o', alpha=0.38,
                    color=colors['harmless + default'])
        ax.scatter(points[indices_with_default_harmless, 0], points[indices_with_default_harmless, 1],
                    marker='o', alpha=0.39,
                    color=colors['harmless + default'], label='harmless + default')

        points = torch.matmul(all_hidden_states_with_mistral_harmless - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_with_mistral_harmless, 0], points[other_indices_with_mistral_harmless, 1],
                    marker='o', alpha=0.38,
                    color=colors['harmless + mistral'])
        ax.scatter(points[indices_with_mistral_harmless, 0], points[indices_with_mistral_harmless, 1],
                    marker='o', alpha=0.39,
                    color=colors['harmless + mistral'], label='harmless + mistral')

        points = torch.matmul(all_hidden_states_with_short_harmless - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_with_short_harmless, 0], points[other_indices_with_short_harmless, 1],
                    marker='o', alpha=0.38,
                    color=colors['harmless + short'])
        ax.scatter(points[indices_with_short_harmless, 0], points[indices_with_short_harmless, 1],
                    marker='o', alpha=0.39,
                    color=colors['harmless + short'], label='harmless + short')

        # harmful
        points = torch.matmul(all_hidden_states - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices, 0], points[other_indices, 1],
                    marker='x', alpha=0.42,
                    color=colors['harmful'])
        ax.scatter(points[indices, 0], points[indices, 1],
                    marker='x', alpha=0.41,
                    color=colors['harmful'], label='harmful')

        points = torch.matmul(all_hidden_states_with_default - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_with_default, 0], points[other_indices_with_default, 1],
                    marker='x', alpha=0.42,
                    color=colors['harmful + default'])
        ax.scatter(points[indices_with_default, 0], points[indices_with_default, 1],
                    marker='x', alpha=0.41,
                    color=colors['harmful + default'], label='harmful + default')

        points = torch.matmul(all_hidden_states_with_mistral - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_with_mistral, 0], points[other_indices_with_mistral, 1],
                    marker='x', alpha=0.42,
                    color=colors['harmful + mistral'])
        ax.scatter(points[indices_with_mistral, 0], points[indices_with_mistral, 1],
                    marker='x', alpha=0.41,
                    color=colors['harmful + mistral'], label='harmful + mistral')

        points = torch.matmul(all_hidden_states_with_short - mean, V)[:, 0:].cpu().numpy()
        ax.scatter(points[other_indices_with_short, 0], points[other_indices_with_short, 1],
                    marker='x', alpha=0.42,
                    color=colors['harmful + short'])
        ax.scatter(points[indices_with_short, 0], points[indices_with_short, 1],
                    marker='x', alpha=0.41,
                    color=colors['harmful + short'], label='harmful + short')


        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if xlim[1] - xlim[0] > ylim[1] - ylim[0]:
            delta = (xlim[1] - xlim[0]) - (ylim[1] - ylim[0])
            ylim = (ylim[0] - delta / 2, ylim[1] + delta / 2)
        else:
            delta = (ylim[1] - ylim[0]) - (xlim[1] - xlim[0])
            xlim = (xlim[0] - delta / 2, xlim[1] + delta / 2)

        with safe_open(f'estimations/{model_name}_all/harmfulness.safetensors', framework='pt') as f:
            weight = torch.mean(f.get_tensor('weight'), dim=0).squeeze(0).tolist()
            bias = torch.mean(f.get_tensor('bias'), dim=0).squeeze(0).tolist()
        boundary_points = calculate_boundary(xlim, ylim, weight, bias)
        logging.info(f"harmfulness boundary: {boundary_points}")
        ax.plot([boundary_points[0][0], boundary_points[1][0]],
                [boundary_points[0][1], boundary_points[1][1]],
                color='black', alpha=1, linewidth=3, linestyle='-.')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if model_name in ['Llama-2-7b-chat-hf', 'vicuna-7b-v1.5', 'CodeLlama-7b-Instruct-hf', 'Mistral-7B-Instruct-v0.2']:
            ax.invert_xaxis()

        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{args.output_path}/{fname}_{args.config}.pdf")

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
