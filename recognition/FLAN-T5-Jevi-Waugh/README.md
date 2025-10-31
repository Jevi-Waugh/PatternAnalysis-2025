#  Fine tuning FLAN-T5 to translate expert radiology reports into layperson summaries  using the BioLaySumm dataset

Author: Jevi Schallum Waugh
## Overview 
This project investigates the full fine tuning of Google's FLAN-T5 models (small and base) for the task of translating expert radiology reports into an easy laman summary using the BioLaySumm 2025 dataset. There were three main approaches although we focus heavily on two of them. We compared Full fine tuning where all parameters were updated vs LoRA adapter which is a parameter efficient fine tuning methoc (PEFT), updating only a small subset of adpater weights. 
The third method is an optimisation technique inspired by biological evolution that optimises model parameters without computing gradients, which will be exploreda and trained, but not in its entirety.

This comparative analysis aims to evaluate the trade-offs between compute and model performance for domain-specific text especifically summarisation texts. 

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithms](#algorithms)
3. [Dataset](#dataset)
4. [Evolution Strategies (ES) Algorithm](#evolution-strategies-es-algorithm)
5. [Models and Architectures](#models-and-architectures)
6. [LoRA Method and Mathematics](#lora-method-and-mathematics)
7. [Fine-Tuning Strategies](#fine-tuning-methods)
8. [Hardware Configuration](#hardware-configuration)
9. [Training Procedure](#training-procedure)
10. [Results and Evaluation](#results-and-evaluation)
11. [Visualisations and Plots](#visualisations-and-plots)
12. [Inference and Prediction Examples](#prediction-examples)
13. [Comparative analysis](#comparative-analysis)
14. [Error Analysis](#error-analysis)
15. [Reproducibility](#reproducibility)
16. [Access to project](#project-access)
17. [References](#references)
---

## Introduction
This project addresses the need to develop automated systems that can translate experts reports into clear, accessible summaries. This is mainly due to the advanced terminologies present in medical documentation that may overwelm the patient.


## Directory Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules (ignores mostly model and checkpoint files)
│
├── dataset.py                         # Dataset loading and preprocessing
├── modules.py                         # Model definitions (FLAN-T5, FLAN-T5-LoRA)
├── train.py                           # Main training point
├── predict.py                         # Inference and prediction script (generating summaries and plots)
│
├── results.txt                        # Raw training logs and evaluation metrics from Google collab
│
├── outputs/                           # All experiment outputs
│   |
│   ├── output_es_flan_base/           # Evolution Strategies + FLAN Base
│   │   ├── training_curves.png
|   |   |__ final_model
│   │   └── checkpoints/
│   │       └── [model checkpoint files]
│   |
│   ├── output_full_finetuning_base/   # Full Fine-Tuning (Base model)
│   │   ├── training_curves.png
|   |   |__ final_model
│   │   └── checkpoints/
│   │       └── [model checkpoint files]
│   |
│   ├── output_full_finetuning_small/  # Full Fine-Tuning (Small model)
│   │   ├── training_curves.png
│   │   └── final_model/
│   |     
│   |
│   ├── output_lora_base/              # LoRA Fine-Tuning (Base model)
│   │   ├── training_curves.png
|   |   |__ final_model
│   │   └── checkpoints/
│   │       └── [model checkpoint files]
│   |
│   ├── output_lora_small/             # LoRA Fine-Tuning (Small model)
│   │   ├── training_curves.png
│   │   └── final_model/
│   |      
│   |
│   └── lora_vs_fft_comparison_3epochs.png  # Comparative plot of LoRA vs FFT
└── 

```
## Algorithms
We compare three fine-tuning paradigms:

1. **Full Fine-Tuning (FFT)**: Traditional fine-tuningapproach updating all ~77M (small) or ~248M (base) parameters.

2. **Low-Rank Adaptation (LoRA)**: Parameter-efficient approach updating only ~1.8M (small) or ~7.1M (base) parameters (~2.2-2.8% of total parameters) (This will be specific in LoRA training in later sections)

The training was initially performed on **Google Colab** with GPU acceleration (Nvidea A100).
## Dataset

### Dataset Statistics

| Split | Examples |
|-------|----------|
| **Train** | 150,454 |
| **Validation** | 10,000 |
| **Test** | 10,537 |
| **Total** | 170,991 |

### Data Structure of the dataset

Each data point contains:
- **`radiology_report`**: Detailed medical report written in professional terminology
- **`layman_report`**: Simplified version of the report for patient understanding
- **`images_path`**: File path to the associated medical image

### Preprocessing and Tokenisation

The preprocessing pipeline has the following steps:

1. **Prompt Engineering**: A task-specific prefix is prepended to each input:
   ```
   "Create a lay summary of this radiology report for a general audience: "
   ```

2. **Tokenisation**: 
   - **Model**: FLAN-T5 tokeniser (`google/flan-t5-small` or `google/flan-t5-base`)
   - **Padding**: Applied to maximum length for efficient batching purposes
   - **Input Length**: Max 512 tokens (truncated if longer)
   - **Target Length**: Max 256 tokens (usually half)

3. **Data Collation**: Dynamic padding using Pytorch's `DataCollatorForSeq2Seq` to optimise memory usage during training

4. **Batching**: 
   - **Small model**: 16 samples per batch
   - **Base model**: 10 samples per batch (due to larger memory)

The preprocessing is very much handled by the `BioDatasetLoader` class in `dataset.py`, which actually provides a clean interface for loading, tokenising, and batching the data. This has a modular design for simplicty purposes. Refer to later sections for the code and intialisation.

---

## Evolution Strategies (ES) Algorithm
### Overview
### What is Evolution Strategies?
### Why is ES being used for LLM Fine-Tuning?
### Implementation Details
### ES Algorithm

## Models and Architectures
### FLAN-T5 Architecture
### Model types
#### 1. FLAN-T5-Small
#### 2. FLAN-T5-Base

### Parameter Comparison: Full Fine-Tuning vs LoRA

### Fine-tuning in models?
#### Full Fine-Tuning (FFT)
#### LoRA Fine-Tuning (PEFT)

## LoRA and Mathematics behind it
### Concept
### Mathematical Formulation
### Forward Pass Computation
### Parameter Reduction
### LoRA details
### LoRA Configuration in This Project for summaries
### Advantages of LoRA
### Tradeoffs and Disadvantages of LoRA



## Fine-Tuning Methods
### Full Fine-Tuning (FFT)
### LoRA Fine-Tuning 
(Parameter-Efficient Fine-Tuning)

## Comparison Table
lOrA vs FFT

## Hyperparameter Settings

### FLAN-T5-Small
### FLAN-T5-Base

## Hardware Configuration
The env it was trained on

## Training 
### Training Time Summary
### Training Procedure

<!-- we put this inside for now -->
### Driver script via Command Line interface
#### Training Arguments
Talk about the primary args and also why bf16 is so crucial and stuff

## Results and Evaluation

### Evaluation Metrics

### Results Summary
#### FLAN-T5-Small
#### FLAN-T5-Base

### Comparative Analysis

#### Strategy Comparison (FLAN-T5-Small, 4 Epochs)
#### Model Size Comparison (LoRA Strategy, 4 Epochs)
#### Training Strategy Comparison (FLAN-T5-Base)


### Winning Model Performance
## Visualisations and Plots

### Plot 1: Evolution Strategies Training (FLAN-T5-Base)

### Plot 2: Full Fine-Tuning Training (FLAN-T5-Base)

### Plot 3: Full Fine-Tuning Training (FLAN-T5-Small)

### Plot 4: LoRA Training (FLAN-T5-Small)

### Plot 5: LoRA Training (FLAN-T5-base)

### Plot 6: LoRA vs Full Fine-Tuning Comparison (3 Epochs)

### Expected Observations

## Prediction Examples
### How to Generate Predictions
### Examples 1,2,3 etc...



## Project Access
### CUDA, GPU and Google Collab
### Installation Instructions
### Dataset Access

## Reproducibility
use my 48829678

### Software Environment
#### Core Dependencies
#### Supporting Libraries

## Error Analysis
### Common Error Patterns in LLM Summarisation

### 1. Hallucinations
### 2. Over-Simplification (SOMETIMES)
### 3. Drift
### 4. Repetition
### 5. Terminology Leakage


## References

### Main Literature

1. **FLAN-T5 Model**:
   - Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Wei, J. (2022). *Scaling instruction-finetuned language models*. arXiv preprint arXiv:2210.11416.
   - [Link to paper](https://arxiv.org/abs/2210.11416)
2. **T5 Original Architecture**:
   - Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer*. Journal of Machine Learning Research, 21(140), 1-67.
   - [Link to paper](https://arxiv.org/abs/1910.10683)
3. **LoRA (Low-Rank Adaptation)**:
   - Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint arXiv:2106.09685.
   - [Link to paper](https://arxiv.org/abs/2106.09685)

4. **Evolution Strategies**:
   - Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). *Evolution strategies as a scalable alternative to reinforcement learning*. arXiv preprint arXiv:1703.03864.
   - [Link to paper](https://arxiv.org/abs/1703.03864)

5. **ROUGE Evaluation Metric**:
   - Lin, C. Y. (2004). *ROUGE: A package for automatic evaluation of summaries*. In Text summarization branches out (pp. 74-81).
   - [Link to paper](https://aclanthology.org/W04-1013/)

---

### Software Documentation

6. **Hugging Face Transformers**:
   - [Transformers Documentation](https://huggingface.co/docs/transformers/)
   - [FLAN-T5 Model Card](https://huggingface.co/google/flan-t5-base)
   - [Summarisation Task Guide](https://huggingface.co/docs/transformers/en/tasks/summarization)

7. **Hugging Face PEFT Library**:
   - [PEFT Documentation](https://huggingface.co/docs/peft/)

8. **Hugging Face Datasets**:
   - [Datasets Documentation](https://huggingface.co/docs/datasets/)
   - [BioLaySumm Dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track)

9. **PyTorch**:
   - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
   - [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

### Domain-Specific Resources

10. **Medical Text Summarisation**:
    - [BioLaySumm 2025](https://biolaysumm.org/)


11. **Parameter-Efficient Fine-Tuning (PEFT) Survey**:
    - Lialin, V., Deshpande, V., & Rumshisky, A. (2023). *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning*. arXiv preprint arXiv:2303.15647.
    - [Link to paper](https://arxiv.org/abs/2303.15647)

---

### Additional References

12. **IBM LoRA Overview**:
    - [IBM - What is LoRA?](https://www.ibm.com/think/topics/lora)

---

### Citation for This Work

If you use this code or methodology in your research, please cite:

```bibtex
@misc{waugh2025flan_t5_lora_biolaysumm,
  author = {Waugh, Jevi},
  title = {Fine-Tuning FLAN-T5 using Full Fine-Tuning and LoRA for BioLaySumm},
  year = {2025},
  publisher = {GitHub},
  howpublished = {}
}
```

---

**End of README**