# modules.py
# Author: Jevi Waugh
# Fine tuning procedure is in this file
from transformers import AutoTokenizer
import torch.nn as nn
from dataset import BioDatasetLoader
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
import os


class FLAN_T5:
    def __init__(self, DataLoader, model_name="google/flan-t5-small"):
        self.dataloader = DataLoader()
        self.dataloader.load_dataset()
        self.tokeniser = self.dataloader._load_tokeniser(model_name)
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.data_collator = None
        
        # Count parameters
        total_params = sum(param.numel() for param in self.model.parameters())
        train_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        print(f"Full fine-tuning: {train_params:,} trainable parameters out of {total_params:,} ({train_params/total_params*100:.2f}%)")
    

    def _padding(self):
        """This is to ensure that we have padding in case it does not set up"""
        if not self.data_collator:
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokeniser,
                model=self.model
            )
    # In case i need them
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokeniser


class FLAN_T5_loRA(FLAN_T5):
    """This class will be used to finetune FLAN-T5 via LoRA

    Args:
        FLAN_T5 (_type_): _description_
    """
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Apply LoRA configuration immediately after initialization
        self._loRA_Config()
    
    def _loRA_Config(self, lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 target_modules=None):
        """Apply LoRA configuration to the model."""
        
        # Apply LoRA configuration
        if target_modules is None:
            # Default: All modules (most capacity, slower)
            target_modules = ["q", "v", "k", "o", "wi", "wo"]
        else: target_modules = ["q", "v"]
        
        lora_config = LoraConfig(
            r=lora_r,    # Make sure to increase the rank for more capacity
            lora_alpha=lora_alpha,  # scaling factor
            target_modules=target_modules,
            lora_dropout=lora_dropout,   # dropout
            bias="none",
            # important for T5
            task_type="SEQ_2_SEQ_LM" 
        )

        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied. Trainable parameters: {self.model.print_trainable_parameters()}")
