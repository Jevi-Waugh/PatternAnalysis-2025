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
from transformers import AutoTokenizer
from dataset import BioDatasetLoader
from modules import FLAN_T5_LoRA, FLAN_T5
import logging
import os
from transformers import AutoModelForSeq2SeqLM
# Configure logger for it to print
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_OUTPUT_DIR = "FLAN-T5-Jevi-Waugh/outputs"

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
        self.SYMBOL = "_"
        
        # need to control batch size for training time flexibiloty
        self.batch_size = batch_size
        self.model = model.to(self.device)
        
        # set prediction generation length
        self.MAX_LENGTH = 256
        
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
        self.schedular = get_linear_schedule_with_warmup(self.optimiser, num_training_steps=len(self.train_dataloader) * num_epochs, num_warmup_steps=0)
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
    
    def print_message(self,message):
        
        aesthetic_bar = 14 + len(message)
        logger.info(f"\n{self.SYMBOL*aesthetic_bar}")
        logger.info(message)
        logger.info(f"\n{self.SYMBOL*aesthetic_bar}")
    
    def train_model(self) -> None:
        """Main training loop for the model. This includes training phase, testing phase through
           the evaluation set and saving the model as well as histories.
        """
        # check checkpoints first
        if self.save_checkpoints: logger.info("The checkpoint will be saved at: {self.checkpoint_dir}")
        
        # loop through epochs
        for epoch in self.num_epochs:
            mss = f"Epoch {epoch + 1}/{self.num_epochs}"
            self.print_message(mss)
            #   get training loss
            training_loss = self._singular_loop(epoch)
            self.history['train_loss'].append(training_loss)
            
            #  Do evaluation and perhaps skip dropout
            self.model.eval()
            t_loss = 0
            preds, all_labels = [],[]
            progress_bar = tqdm(self.eval_dataloader, desc="Evaluation Phase")
            
            with torch.no_grad():
                # get data ready
                batch = {}
                # loop through each batch
                for batch in progress_bar:
                    for num, sample in batch.items():
                        # Transfer to device - GPU
                        batch[num] = sample.to(self.device)
                    # get loss
                    with torch.cuda.amp.autocast("cuda", dtype=torch.float32):
                        outputs = self.model(**batch)
                    
                    t_loss += outputs.loss.item()
                    
                    # generate predictions
                    inputs = batch["input_ids"]
                    attention_m  = batch["attention_mask"]
                    tokens = self.model.generate(inputs, attention_m, max_length=self.MAX_LENGTH)
                    
                    # use tokeiser to decode
                    decoded_preds = self.tokeniser.batch_decode(tokens, skip_special_tokens=True)
                    
                    # get labels
                    labels = batch["labels"]
                    # decode label
                    pad_tok = self.tokeniser.pad_token_id
                    labels = torch.where(labels != -100, labels, pad_tok)
                    decoded_labels = self.tokeniser.batch_decode(labels, skip_special_tokens=True)
                    
                    preds.extend(decoded_preds)
                    all_labels.extend(decoded_labels)
            # compute rouge scores
            rouge_scores = self.rouge.compute(predictions=preds,references=all_labels,use_stemmer=True)            
            # gen length - get the mean
            each_pred = []
            for pred in preds:
                each_pred.append(pred.split())
            avg_gen_length = np.mean(each_pred)
            # in case i need to return it
            eval_metrics = {
                'eval_loss': t_loss / len(self.eval_dataloader),
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeLsum': rouge_scores['rougeLsum'],
                'rougeL': rouge_scores['rougeL'],
                'gen_len': avg_gen_length
            }
            # Save scores into history
            self.history['eval_loss'].append(eval_metrics['eval_loss'])
            self.history['rouge1'].append(eval_metrics['rouge1'])
            self.history['rouge2'].append(eval_metrics['rouge2'])
            self.history['rougeLsum'].append(eval_metrics['rougeLsum'])
            self.history['rougeL'].append(eval_metrics['rougeL'])
            self.history['gen_len'].append(eval_metrics['gen_len'])
            
            # print results
            self.print_message(message=f"Epoch: {epoch + 1} outputs.")
            print(f"  Training Loss:   {training_loss:.4f}")
            print(f"  Validation Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"  ROUGE-1:         {eval_metrics['rouge1']:.4f}")
            print(f"  ROUGE-2:         {eval_metrics['rouge2']:.4f}")
            print(f"  ROUGE-Lsum:      {eval_metrics['rougeLsum']:.4f}")
            print(f"  ROUGE-L:         {eval_metrics['rougeL']:.4f}")
            print(f"  Gen Length:      {eval_metrics['gen_len']:.2f}")
            
            # save checkpoints
            if self.save_checkpoints: self._save_checkpoint(epoch + 1, eval_metrics)
            
        # save history and metrics
        self._save_history()
        
    def _singular_loop(self, epoch):
        """This will train for a only a singular epoch passing through teh entire dataset.
        """

        class EpochTracker:
            """Tracker for epoch
            """
            def __init__(self):
                self.epoch_losses : list[float] = []
                self.total: int = 0
                
            def add_loss(self, loss):
                """This will aggregate each loss into the total

                Args:
                    loss (_type_): Aggregated loss
                """
                self.epoch_losses.append(loss)
                self.total += loss
                
            def get_avg(self): 
                """Returns the average loss per epoch

                Returns:
                    _type_: Average loss
                """
                return self.total / len(self.epoch_losses) is self.epoch_losses else 0
        
        self.model.train()
        progress_bar = tqdm(self.train_dataloader, desc=f"Training epochs: {epoch+1}")
        tracker = EpochTracker()
        # get data ready
        batch = {}
        for batch in progress_bar:
            for num, sample in batch.items():
                batch[num] = sample.to(self.device)
        
        # got foward and backwards at the same time
        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = self.model(batch)
            loss = result.loss
            
        # get a scalar to backpropagate the gradients first
        # ensure that the computation graph will be reset afterwards
        
        self.scalar.scale(loss).backward()
        self.scalar.step(self.optimiser)
        
        self.scalar.update()
        
        # update and optimise
        self.optimiser.zero_grad()
        self.schedular.step()
        
        # keep tracking
        loss_estimate = loss.item()
        tracker.add_loss(loss_estimate)
        simp_loss = f"{loss_estimate:.4f}"
        loss_dict = {"LOSS": simp_loss}
        progress_bar.set_postfix(loss_dict)
        
        return tracker.get_avg()
    
class ESTrainer():
    """Evolution Strategies trainer for FLAN-T5
    """
    def __init__(self, model_name="google/flan-t5-base", output_directory=None):
        self.model_name = model_name
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        # Disabling dropout for deterministic evaluation
        self.model.eval()  
        
        # Load ROUGE metric
        self.rouge = evaluate.load("rouge")
        
        # Setup output directory
        if output_directory is None:
            output_directory = os.path.join(BASE_OUTPUT_DIR, "output_es_flan_base")
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(os.path.join(output_directory, "checkpoints"), exist_ok=True)
    
    #  Training history
        self.history = {
            'mean_reward': [],
            'min_reward': [],
            'max_reward': [],
            'std_reward': []
        }
        
        print(f"The Model has been loaded on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Output directory: {output_directory}")

    def _save_history(self):
        """Save training history and corresponding metrics used to evaluate the model.
        """
        h_path = os.path.join(self.output_directory, "training_history.json")
        # Open file and dump all metrics
        with open(h_path, 'w') as f:
            json.dump(self.saved_history, f, indent=3)
        logger.info(f"Training history has successfully been saved to: {h_path}")
        
    
    def save_checkpoint(self, iteration):
        """Save model checkpoint and training info"""
        checkpoint_dir = os.path.join(
            self.output_directory, 
            "checkpoints", 
            f"checkpoint-iteration-{iteration}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokeiser
        self.model.save_pretrained(checkpoint_dir)
        self.tokeniser.save_pretrained(checkpoint_dir)
        
        # Save checkpoint info
        checkpoint_info = {
            'iteration': iteration,
            'mean_reward': self.history['mean_reward'][-1],
            'max_reward': self.history['max_reward'][-1],
            'min_reward': self.history['min_reward'][-1],
            'std_reward': self.history['std_reward'][-1]
        }
        
        with open(os.path.join(checkpoint_dir, "checkpoint_info.json"), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint saved at: {checkpoint_dir}")