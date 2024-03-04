# LLM-Safeguard

Official repository for paper "[On Prompt-Driven Safeguarding for Large Language Models](https://arxiv.org/abs/2401.18018)"

If you find this repository useful or our work is related to your research, please kindly cite it:

```latex
@article{
  llm-safeguard,
  title={On Prompt-Driven Safeguarding for Large Language Models},
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

## Overview

As we know, prepending model inputs with **safety prompts** is a common means for safeguarding large language models (LLMs) from complying with harmful queries. This practice has been adopted in real-world deployed LLMs like ChatGPT and Mistral.

<img src="README.assets/problem_example.png" alt="problem_example" style="zoom:30%;" style="display: block; margin-left: auto; margin-right: auto;"/>

Do you know **how safety prompts intrinsically work in LLM safeguarding**? Our work reveals their working mechanisms from the perspective of **model representations**, and opens the potential for automatically optimizing them to improve LLM safety.

We find that in models’ representation space, harmful and harmless queries can be largely distinguished, but this is not noticeably enhanced by safety prompts (the below figure **upper**). Instead, the queries’ representations are moved by safety prompts in similar directions where models become more prone to refusing to provide assistance (the below figure **lower**) even when the queries are harmless.

<img src="README.assets/visualization.png" alt="visualization" style="zoom:30%;" style="display: block; margin-left: auto; margin-right: auto;"/>

Inspired by these findings, we propose a method called **DRO (Directed Representation Optimization)** for automatic safety prompt optimization. It treats safety prompts as continuous, trainable embeddings and learns to **move the representations of harmful/harmless queries along/opposite the direction in which the model’s refusal probability increases**. 

<img src="README.assets/visualization_post.png" alt="visualization_post" style="zoom:30%;" style="display: block; margin-left: auto; margin-right: auto;"/>

Experiments with eight LLMs on out-of-domain benchmarks demonstrate that DRO remarkably improves the safeguarding performance of human-crafted safety prompts and outperforms strong baselines, without compromising the general model capability.

<img src="README.assets/results.png" alt="results" style="zoom:30%;" style="display: block; margin-left: auto; margin-right: auto;"/>

Please refer to our paper for the technical details of DRO.

## Code and Data

See [`code`](code/) for the experimental code for reproducing our experimental results.

We also release the experimental data and results in another repo: https://github.com/chujiezheng/LLM-Safeguard_data
