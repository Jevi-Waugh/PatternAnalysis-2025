#  Fine tuning FLAN-T5 to translate expert radiology reports into layperson summaries  using the BioLaySumm dataset

Author: Jevi Schallum Waugh
## Overview 
This project investigates the full fine tuning of Google's FLAN-T5 models (small and base) for the task of translating expert radiology reports into an easy laman summary using the BioLaySumm 2025 dataset. There were three main approaches although we focus heavily on two of them. We compared Full fine tuning where all parameters were updated vs LoRA adapter which is a parameter efficient fine tuning methoc (PEFT), updating only a small subset of adpater weights. The third method is an optimization technique inspired by biological evolution that optimizes model parameters without computing gradients. 

This comparative analysis aims to evaluate the trade-offs between compute and model performance for domain-specific text especifically summarisation texts. 

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

# References
* Hugging Face
https://huggingface.co/docs/transformers/en/tasks/summarization
*IBM
https://www.ibm.com/think/topics/lora