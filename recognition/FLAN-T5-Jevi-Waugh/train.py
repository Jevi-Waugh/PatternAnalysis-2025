# train.py - Training, validation, testing and saving
# Author: Jevi Waugh

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os
import json
import evaluate
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import BioDatasetLoader
from modules import FLAN_T5_LoRA, FLAN_T5_FullFineTuning


class FlanT5_Trainer:
    """Optimised Trainer for FLAN Models"""
    
    def __init__(self, model, tokenizer, train_dataset, test_dataset, data_collator,
                 num_epochs=4, batch_size=16, learning_rate=3e-4,
                 output_dir="./output", checkpoints_saving=True, device=None):
        
        # Use GPU if possible
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Default parameters for the model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.checkpoints_saving = checkpoints_saving
        
        # Making sure that the directory is created
        os.makedirs(output_dir, exist_ok=True)
        # If we have checkpoints then save them
        if checkpoints_saving:
            # Because we want to have a dedicated section so that we don't lose progress
            self.checkpoints_saving = os.path.join(self.output_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Create dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )
        
        # Seting yp the optimizer
        # This has to return AutoModelForSeq2SeqLM.from_pretrained(model_name).parameters()
        # remember
        # set decay to 0.01 for now
        self.optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)