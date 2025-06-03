# OWS (OwLore)

This repo contains the pre-release version of OWS algorithm, proposed by [Outlier-weighed Layerwise Sampling for LLM Fine-tuning](https://arxiv.org/abs/2405.18380).

Outlier-weighed Layerwise Sampling for LLM Fine-tuning is a novel memory-efficient LLM fine-tuning approach, enhances fine-tuning performance by using layerwise sampling and gradient low-rank training.

<div align="center">
  <img src="https://github.com/pixeli99/OwLore/assets/46072190/fb60054b-7af1-4aa0-9cc8-329c0f96d093" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>

## Abstract

The rapid advancements in Large Language Models (LLMs) have revolutionized various natural-language-processing tasks. However, the substantial size of LLMs presents significant challenges for training or fine-tuning. While parameter-efficient approaches such as low-rank adaptation (LoRA) have gained popularity, they often compromise performance compared with full-rank fine-tuning.
In this paper, we propose Outlier-weighed Layerwise Sampling (OWS), a new memory-efficient fine-tuning approach inspired by the layer-wise outlier distribution of LLMs. Unlike LoRA—which adds extra adapters to all layers—OWS strategically assigns higher sampling probabilities to layers with more outliers, selectively sampling only a few layers and fine-tuning their pre-trained weights. To further increase the number of fine-tuned layers without a proportional rise in memory cost, we incorporate gradient low-rank projection, boosting performance even more.
Extensive experiments on architectures including LLaMA 2 and Mistral show that OWS consistently outperforms baseline approaches, even full fine-tuning. Specifically, OWS achieves up to a 1.1 % average accuracy gain on the Commonsense Reasoning benchmark, 3.0 % on MMLU, and a notable 10 % boost on MT-Bench, all while being more memory-efficient. OWS enables fine-tuning 7 B-parameter LLMs with only 21 GB of memory.

## Quick Start

### Setup

Our repository is built on top of [LMFlow](https://github.com/OptimalScale/LMFlow). You can configure the environment using the following command lines:
```bash
conda create -n owlore python=3.9 -y
conda activate owlore
conda install mpi4py
bash install.sh
pip install peft
```

### Prepare Dataset

You can download our processed datasets from Hugging Face [here](https://huggingface.co/datasets/pengxiang/OwLore_Dataset).

### Finetuning Examples

--- 
We provide a quick overview of the arguments:
- `--model_name_or_path`: The identifier for the model on the Hugging Face model hub.
- `--lisa_activated_layers`: Specifies the number of layers to activate at each step during training.
- `--lisa_interval_steps`: Indicates the number of steps after which resampling occurs.
- `--lisa_prob_mode`: Defines the method used to determine the sampling probability, which can include options such as `uniform`, `owl`, `decrease`, `increase`, etc.
- `--galore`: Indicates whether to use GaLore as the optimizer.

#### Commonsense Reasoning
The script will run LISA on the `Commonsense Reasoning` dataset.
```bash
bash owlore_scripts/run_lisa.sh merge # LISA
```
The script will run OwLore on the `Commonsense Reasoning` dataset.
```bash
bash owlore_scripts/run_owlore_low_rank.sh merge # OwLore
```

#### MMLU
The script will run LISA on the `MMLU` dataset.
```bash
bash owlore_scripts/run_lisa.sh mmlu # LISA
```
The script will run OwLore on the `MMLU` dataset.
```bash
bash owlore_scripts/run_owlore_low_rank.sh mmlu # OwLore
```

#### GSM8K
The script will run LISA on the `GSM8k` dataset.
```bash
bash owlore_scripts/run_lisa.sh gsm # LISA
```
The script will run OwLore on the `GSM8k` dataset.
```bash
bash owlore_scripts/run_owlore_low_rank.sh gsm # OwLore
```

### Evaluation

We use [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to obtain evaluation results. Please refer to its installation instructions to configure `lm_eval`. The steps are as follows:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

After setting up the environment, use the following command to run the evaluation:

#### MMLU
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B\
    --tasks mmlu \
    --output_path mmlu_results \
    --num_fewshot 5 \
    --batch_size auto \
    --cache_requests true
```
#### Commonsense Reasoning
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B\
    --tasks boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
    --output_path qa_results \
    --num_fewshot 5 \
    --batch_size auto \
    --cache_requests true
```
#### GSM8K
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B\
    --tasks gsm8k \
    --output_path math_results \
    --batch_size auto \
    --cache_requests true
```

### Acknowledgement
This repository is build upon the [LMFlow](https://github.com/OptimalScale/LMFlow) and [OWL](https://github.com/luuyin/OWL) repositories. Thanks for their great work!

## Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```
@misc{li2024owlore,
      title={Outlier-weighed Layerwise Sampling for LLM Fine-tuning}, 
      author={Pengxiang Li and Lu Yin and Xiaowei Gao and Shiwei Liu},
      year={2024},
      eprint={2405.18380},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
