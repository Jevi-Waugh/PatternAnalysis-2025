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


class BioTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, data_collator,
                 learning_rate=5e-4, num_epochs=4, batch_size=16, 
                 output_dir="./output", device=None, save_checkpoints=True):
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        # in case training fails midway through epoch
        self.save_checkpoints = save_checkpoints
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        
        # need to control batch size for training time flexibiloty
        self.batch_size = batch_size
        self.model = model.to(self.device)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        if save_checkpoints: 
            self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Create dataloaders for both training and evaluationn sets
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        # Setup optimizer, look if there is another pne better than adam for fine tuning
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Figure out how to set uop schedulaer later
        
        # rouge
        self.rouge = evaluate.load("rouge")
        
        self.saved_history = {
            'train_loss': [],
            'eval_loss': [],
            'rouge1': [],
            'rouge2': [],
            'rougeLsum': [],
            'rougeL': [],
            # Make sure that the generation length is consistent
            'gen_len': []
        }
        
    def train(self):
        pass
    
    def save_model(self):
        pass
    
    def _save_checpoint(self):
        pass