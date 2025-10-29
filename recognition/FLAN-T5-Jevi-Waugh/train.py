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
from modules import FLAN_T5_LoRA, FLAN_T5
import logging
import os
logger = logging.getLogger(__name__)

class BioTrainer:
    def __init__(self, model, tokeniser, train_dataset, eval_dataset, data_collator,
                 learning_rate=5e-4, num_epochs=4, batch_size=16, 
                 output_dir="./output", device=None, save_checkpoints=True):
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokeniser = tokeniser
        # in case training fails midway through epoch
        self.save_checkpoints = save_checkpoints
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        
        # need to control batch size for training time flexibiloty
        self.batch_size = batch_size
        self.model = model.to(self.device)
        
        os.makedirs(output_dir, exist_ok=True)
        if save_checkpoints: 
            self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Create dataloaders for both training and evaluationn sets
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        # Setup optimiser, look if there is another pne better than adam for fine tuning
        self.optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Figure out how to set up scheduler
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(cuda=True, enabled=True)
        # rouge
        self.rouge = evaluate.load("rouge")
        
        # might need that to plot later
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
        
    
    def save_model(self, saving_path=None):
        """Saves the model and tokeniser 

        Args:
            saving_path (_type_, optional): Path to save the model. Defaults to None.
        """
        if saving_path is None: saving_path = os.path.join(self.output_dir, "final_model")
        os.makedirs(saving_path, exist_ok=True)
        self.model.save_pretrained(saving_path)
        self.tokeniser.save_pretrained(saving_path)
        logger.info(f"Model saved to: {saving_path}")
        
    def _save_history(self):
        """Save training history and corresponding metrics used to evaluate the model.
        """
        h_path = os.path.join(self.output_dir, "training_history.json")
        # Open file and dump all metrics
        with open(h_path, 'w') as f:
            json.dump(self.saved_history, f, indent=3)
        logger.info(f"Training history has successfully been saved to: {h_path}")
        
    def _save_checkpoint(self, epoch, eval_results):
        """This function will save the model checkpoints after each epoch.

        Args:
            epoch (_type_): Current epoch
            eval_results (_type_): Evaluation results after that epoch
        """
        # Always making sure that the folder path is set up
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch-checkpoint-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Saving the tokeniser and modelr
        self.model.save_pretrained(checkpoint_path)
        self.tokeniser.save_pretrained(checkpoint_path)
        
        # Save checkpoint info
        checkpoint_metrics = {
            # training and eval loss
            'train_loss': self.history['train_loss'][-1],
            'eval_loss': eval_results['eval_loss'],
            # ROUGE-1 is the overlap of unigram (single word) matches between the candidate and reference.
            'rouge1': eval_results['rouge1'],
            # ROUGE-L: is the longest common subsequence between the candidate and reference.
            'rougeL': eval_results['rougeL'],
            # ROUGE-2 is the overlap of bigram (two-word sequence) matches.
            'rouge2': eval_results['rouge2'],
            # ROUGE-Lsum is the sentence-level ROUGE-L averaged across the whole summary.
            'rougeLsum': eval_results['rougeLsum'],
            # gen_len is the average generation length token of the summary itself.
            'gen_len': eval_results['gen_len'],
            # current epoch
            'epoch': epoch
        }
        
        checkpoint_metrics_path = os.path.join(checkpoint_path, "checkpoint_metrics.json")
        with open(checkpoint_metrics_path, 'w') as f: json.dump(checkpoint_metrics, f, indent=3)
        logger.info(f" Checkpoint saved at: {checkpoint_path}")
        
    def train_model(self):
        """Main training loop for the model. This includes training phase, testing phase through
           the evaluation set and saving the model as well as histories.
        """
        pass
        
    def _singular_loop(self):
        """This will train for a only a singular epoch passing through teh entire dataset.
        """
        
        class EpochTracker:
            def __init__(self):
                self.epoch_losses : list[float] = []
                self.total: int = 0
                
            def add_loss(self, loss):
                self.epoch_losses.append(loss)
                self.total += loss
                
            def get_avg(self): 
                return self.total / len(self.epoch_losses) is self.epoch_losses else 0
        
        self.model.train()
        tracker = EpochTracker()
        # get data ready
        batch = {}
        for b in self.train_dataloader:
            for num, sample in b.items():
                batch[num] = sample.to(self.device)
        
        # got foward and backwards at the same time
        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = self.model(batch)
            loss = result.loss
            
        # get a scalar to backpropogate the gradients first
        # ensure tyhat the computation graph will be reset afterwards
        
        self.scalar.scale(loss).backward()
        self.scalar.step(self.optimiser)
        
        # update and optimise
        self.optimiser.zero_grad()
        
        
        # keep tracking
        tracker.add_loss(loss.item())
        
    
    def _backward_pass():
        pass