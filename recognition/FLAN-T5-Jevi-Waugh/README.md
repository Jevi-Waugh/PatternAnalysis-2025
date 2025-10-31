#  Fine tuning FLAN-T5 to translate expert radiology reports into layperson summaries  using the BioLaySumm dataset

Author: Jevi Schallum Waugh
## Overview 
This project investigates the full fine tuning of Google's FLAN-T5 models (small and base) for the task of translating expert radiology reports into an easy laman summary using the BioLaySumm 2025 dataset. There were three main approaches although we focus heavily on two of them. We compared Full fine tuning where all parameters were updated vs LoRA adapter which is a parameter efficient fine tuning methoc (PEFT), updating only a small subset of adpater weights. The third method is an optimization technique inspired by biological evolution that optimizes model parameters without computing gradients. 

This comparative analysis aims to evaluate the trade-offs between compute and model performance for domain-specific text especifically summarisation texts. 

## Table of Contents
---

## Introduction

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

## Dataset

### Dataset Information

### Dataset Statistics

### Data Structure

### Preprocessing and Tokenisation

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
## Visualizations and Plots

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


## Error Analysis
### Common Error Patterns in LLM Summarisation

### 1. Hallucinations
### 2. Over-Simplification (SOMETIMES)
### 3. Drift
### 4. Repetition
### 5. Terminology Leakage


## Reproducibility
use my 48829678


### Software Environment

#### Core Dependencies
#### Supporting Libraries














# References
* Hugging Face
https://huggingface.co/docs/transformers/en/tasks/summarization
*IBM
https://www.ibm.com/think/topics/lora