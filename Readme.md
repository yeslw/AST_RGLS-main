# AST-RGLS: Adaptive Structural Transition and Residual-Guided Local Swap for LLM Pruning

This repository provides the implementation of **AST-RGLS**, a training-free post-training pruning framework for large language models. The project is adapted from the RIA codebase and further improves the pruning pipeline through sparsity-adaptive importance reweighting and residual-guided mask refinement.

## Overview

Post-training pruning is an efficient way to compress large language models without full retraining. Existing calibration-based pruning methods usually build masks with fixed importance criteria. However, the reliability of row-wise and column-wise importance signals changes as the target sparsity increases, especially under aggressive compression.

AST-RGLS addresses this problem with two components:

* **Adaptive Structural Transition (AST):** adaptively rebalances row-wise and column-wise structural importance according to the target sparsity ratio.
* **Residual-Guided Local Swap (RGLS):** refines the initial pruning mask by swapping uncertain weights near the pruning threshold while strictly preserving the target sparsity.

The method is designed for **training-free unstructured post-training pruning**. It only requires calibration forward passes and does not perform gradient-based fine-tuning or weight updates.

## Main Features

* Training-free post-training pruning for LLM compression
* Compatible with RIA-style calibration and pruning pipeline
* Sparsity-aware importance estimation through AST
* One-shot local mask refinement through RGLS
* Support for common pruning baselines such as Magnitude, Wanda, SparseGPT, and RIA
* Evaluation on WikiText-2 perplexity and zero-shot downstream tasks

## Method

The pruning pipeline contains four main steps:

1. **Calibration activation collection**

   A small calibration set is passed through the dense model to collect layer-wise activation statistics.

2. **AST-based importance estimation**

   AST computes an importance score by combining activation-aware weight importance with row-wise and column-wise structural information. Different from fixed scoring rules, AST uses a sparsity-dependent coefficient to shift the importance balance from column-dominant to row-dominant as sparsity increases.

3. **Initial mask construction**

   For each prunable layer, weights with higher AST scores are retained and low-score weights are pruned according to the target sparsity ratio.

4. **RGLS mask refinement**

   RGLS focuses on weights close to the pruning threshold. It builds a drop pool from barely retained weights and an add pool from barely pruned weights, then performs the same number of drop-add swaps based on residual-guided benefit scores. This improves the mask while keeping the sparsity unchanged.

## Installation

Create a conda environment:

```bash
conda create -n ast-rgls python=3.10
conda activate ast-rgls
```

Install dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

If zero-shot evaluation is needed, install `lm-evaluation-harness`:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## Usage

### Unstructured pruning with AST-RGLS

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method ast_rgls \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save
```

### Run baseline methods

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method ria \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save
```

The `--prune_method` argument can be replaced with:

```text
magnitude | wanda | sparsegpt | ria | ast | ria_rgls | ast_rgls
```

The exact available method names depend on the registration in `main.py`.

### Zero-shot evaluation

```bash
python main.py \
  --model YOUR_MODEL_NAME_OR_PATH \
  --prune_method ast_rgls \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --eval_zero_shot
```

## Experimental Results

### WikiText-2 perplexity under 50% unstructured sparsity

Lower perplexity is better.

| Method    | OPT-1.3B | OPT-2.7B | OPT-6.7B | LLaMA-7B | LLaMA-13B | LLaMA-2-7B | LLaMA-2-13B |
| --------- | -------: | -------: | -------: | -------: | --------: | ---------: | ----------: |
| Wanda     |    19.42 |    15.32 |    16.03 |     7.93 |      6.45 |       7.64 |        6.34 |
| SparseGPT |    18.33 |    14.26 |    12.32 |     7.31 |      6.53 |       7.70 |        6.71 |
| RIA       |    19.20 |    14.11 |    12.08 |     7.13 |      6.40 |       7.58 |        6.32 |
| AST-RGLS  |    18.39 |    13.98 |    11.74 |     7.08 |      6.22 |       7.22 |        6.12 |

### Zero-shot accuracy on LLaMA-2-7B under 50% unstructured sparsity

Higher accuracy is better.

| Method    | HellaSwag | BoolQ | WinoGrande |  MNLI |  WNLI | Average |
| --------- | --------: | ----: | ---------: | ----: | ----: | ------: |
| Magnitude |     53.03 | 67.29 |      67.12 | 36.63 | 42.39 |   53.29 |
| Wanda     |     56.64 | 80.84 |      72.84 | 43.81 | 44.84 |   59.79 |
| SparseGPT |     54.47 | 79.37 |      71.35 | 43.17 | 46.26 |   58.92 |
| RIA       |     56.13 | 78.64 |      71.84 | 44.44 | 46.14 |   59.44 |
| AST-RGLS  |     56.89 | 79.46 |      72.02 | 44.84 | 47.53 |   60.15 |

### Component ablation on LLaMA-7B

WikiText-2 perplexity under 60% and 70% unstructured sparsity.

| Method     | 60% Sparsity | 70% Sparsity |
| ---------- | -----------: | -----------: |
| RIA        |        10.56 |        93.69 |
| AST        |        10.02 |        76.87 |
| RIA + RGLS |        10.22 |        81.03 |
| AST + RGLS |         9.54 |        69.95 |

These results show that AST improves the global importance ranking, while RGLS further corrects local boundary errors. The combined method is especially useful under high sparsity.

## Project Structure

```text
.
├── main.py                 # Entry point for pruning and evaluation
├── lib/                    # Pruning, evaluation, and utility modules
├── requirements.txt        # Python dependencies
├── scripts/                # Optional running scripts
└── README.md
```

The exact directory structure may vary depending on the local code organization.

## Notes

* The method mainly targets **unstructured post-training pruning**.
* Calibration data quality can affect pruning stability.
* RGLS performs one-shot mask refinement and does not update model weights.
* For large models, GPU memory usage depends on model size, calibration sequence length, and batch size.

## Acknowledgement

This project is developed based on the RIA repository and follows its pruning pipeline. The original RIA repository is built upon SparseGPT and Wanda. We thank the authors of RIA, SparseGPT, and Wanda for their open-source contributions.

## Citation

If you use this codebase, please cite the original RIA paper:

```bibtex
@inproceedings{
zhang2024plugandplay,
title={Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models},
author={Yingtao Zhang and Haoli Bai and Haokun Lin and Jialin Zhao and Lu Hou and Carlo Vittorio Cannistraci},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Tr0lPx9woF}
}
```

The citation information for AST-RGLS will be updated after publication.
