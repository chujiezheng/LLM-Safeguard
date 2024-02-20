# README

You can find the scripts of running LLMs with human-crafted safety prompts and training continuous safety prompts in `scripts`. Note that for local running you should set the env variable `HF_MODELS` that indicates the save folder of LLMs.

If you find this repository useful or our work is related to your research, please kindly cite it:

```latex
@article{
  llm-safeguard,
  title={Prompt-Driven LLM Safeguarding via Directed Representation Optimization},
  author={Chujie Zheng and Fan Yin and Hao Zhou and Fandong Meng and Jie Zhou and Kai-Wei Chang and Minlie Huang and Nanyun Peng},
  journal={arXiv preprint arXiv:2401.18018},
  year={2024}
}
```

If you find the chat templates used in this project useful, please also kindly cite it:

```latex
@misc{zheng-2023-chat-templates,
  author = {Zheng, Chujie},
  title = {Chat Templates for HuggingFace Large Language Models},
  year = {2023},
  howpublished = {\url{https://github.com/chujiezheng/chat_templates}}
}
```

## How to Run Code

To get generation and evaluation results with **human-crafted safety prompts**, run:

```sh
bash scripts/run_mistral-v1.sh
bash scripts/run_mistral-v1_harmless.sh
```

To train **continuous safety prompts**, and then get generation and evaluation results, run:

```sh
bash scripts/forward.sh
bash scripts/forward_harmless.sh
bash scripts/train_mistral-v1.sh
```

You may uncomment the *unlikelihood* line to reproduce the *vanilla Prompt Tuning* baseline.

To **visualize the hidden states with estimated boundaries**, run:

```sh
bash scripts/compare_gather.sh
```

## Experimental Results

Our experimental results are released in another data repo: https://github.com/chujiezheng/LLM-Safeguard_data 

## Acknowledgement

Our code base builds upon the follow repository: https://github.com/Princeton-SysML/Jailbreak_LLM
