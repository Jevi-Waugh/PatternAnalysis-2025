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
import time
import gc
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
        # plot according to spec
        self.plot_training_curves()
        
    def plot_training_curves(self):
        """Plottings and saving training curves for BioTrainer"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-o', label='Training Loss')
        axes[0, 0].plot(epochs, self.history['eval_loss'], 'r-o', label='Validation Loss')
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Generation length
        axes[1, 0].plot(epochs, self.history['gen_len'], 'm-o')
        axes[1, 0].grid(True)
        axes[1, 0].set_title('Generation Length Over Training')
        axes[1, 0].set_ylabel('Average Generation Length')
        axes[1, 0].set_xlabel('Epoch')
        
        # ROUGE-Lsum
        axes[1, 1].plot(epochs, self.history['rougeLsum'], 'c-o')
        axes[1, 1].grid(True)
        axes[1, 1].set_ylabel('ROUGE-Lsum Score')
        axes[1, 1].set_title('ROUGE-Lsum Over Training')
        axes[1, 1].set_xlabel('Epoch')
        
        # ROUGE scores
        axes[0, 1].plot(epochs, self.history['rouge1'], 'g-o', label='ROUGE-1')
        axes[0, 1].plot(epochs, self.history['rouge2'], 'b-o', label='ROUGE-2')
        axes[0, 1].plot(epochs, self.history['rougeL'], 'r-o', label='ROUGE-L')
        axes[0, 1].set_title('ROUGE Scores Over Training')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROUGE Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
        # show while training
        plt.show()
        plt.close()
        
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
        
        # ES Hyperparameters (from paper - Table 1)
        # Defined such param for short compute
        self.POPULATION_SIZE = 10  # This is the Number of perturbed models per ITERATION
        self.SIGMA = 0.001         # Noise scale (standard deviation)
        self.ALPHA = 0.0005        # Learning rate
        self.NUM_ITERATIONS = 15   # Number of ES iterations
        self.VAL_SAMPLES = 30      # Subset of validation set for evaluation

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
        
    
    def perturb_weights(self, sigma, restore=False, seed=48829678):
        """
        Add or subtract Gaussian noise to all model parameters. This is a variant of process_seed()
        from the EVOLUTION STRATEGIES AT SCALE: LLM FINETUNING BEYOND REINFORCEMENT LEARNING Paper
        
        Args:
            seed: Random seed for reproducible perturbation
            sigma: Noise scale (standard deviation)
            restore: If True, subtract noise; if False, add noise
        """
        sign = -1.0 if restore else 1.0
        
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            
            # Create a generator
            gen = torch.Generator(device=param.device)
            # get the seed
            gen.manual_seed(int(seed))
            
            # Generate some nice noise
            noise = torch.randn(
                param.shape,
                dtype=param.dtype,
                device=param.device,
                generator=gen
            )
            
            param.data.add_(sign * sigma * noise)
            # for optimisation purposes we delete the noise manually from memory
            del noise
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
    def es_update(self, seeds, rewards):
        """
        Apply ES update: aggregate perturbations weighted by normalised rewards.
        
        This implements the core ES algorithm from the paper (Algorithm 2).
        """
        # Normalize rewards (z-score)
        rewards = np.array(rewards)
        rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Aggregate weighted perturbations
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            
            update = torch.zeros_like(param)
            
            # Generate noise
            for seed, reward_norm in zip(seeds, rewards_norm):
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                
                noise = torch.randn(
                    param.shape,
                    dtype=param.dtype,
                    device=param.device,
                    generator=gen
                )
                
                update.add_(reward_norm * noise)
                del noise
            
            # According to the paper, we
            # Update lth layer’s parameters in-place as 
            # shown below
            # θ ← θ + α * (1/N) *  Σ( R_normalised * ε )
            param.data.add_((self.ALPHA / self.POPULATION_SIZE) * update)
            del update
            
        torch.cuda.empty_cache()
        
    
    def evaluate_population_member(self, val_data, seed, val_samples):
        """
        Evaluate a single perturbed model on validation set.
        
        Returns:
            reward: ROUGE-1 score (used as fitness/reward)
        """
        # Add some noise
        self.perturb_weights(seed, self.SIGMA, restore=False)
        
        # Generate summaries on subset
        predictions, references = [], []
        
        # Use subset for faster evaluation
        eval_indices = np.random.choice(len(val_data), 
                                       min(val_samples, len(val_data)), 
                                       replace=False)
        
        for idx in eval_indices:
            sample = val_data[int(idx)]
            
            # Prepare input
            prefix = "Create a lay summary of this radiology report for a general audience:: "
            input_text = prefix + sample['radiology_report']
            
            # set up tokeniser
            inputs = self.tokeniser(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad(), torch.cuda.amp.autocast("cuda", dtype=torch.bfloat32):
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=1,  # Greedy decoding for ES (deterministic)
                )
            
            pred = self.tokeniser.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append(sample['layman_report'])
        
        # Compute ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # Use ROUGE-1 as reward (primary metric we care about)
        reward = rouge_scores['rouge1']
        
        # Restore original weights (subtract noise)
        self.perturb_weights(seed, self.SIGMA, restore=True)
        
        # Cleanup for optimisation
        del inputs, outputs, predictions, references
        torch.cuda.empty_cache()
        
        return reward
    
    def train(self, val_data, num_iterations=10, save_every=100, initial_seed=42) -> AutoModelForSeq2SeqLM:
        """
        This is the main ES training loop.
        
        Args:
            val_data: Validation dataset
            num_iterations: Number of ES iterations
            save_every: Save checkpoint every N iterations
            initial_seed: Random seed for reproducibility
        """
        print(f"Population size: {self.POPULATION_SIZE}")
        print(f"Noise scale (σ): {self.SIGMA}")
        print(f"Learning rate (α): {self.ALPHA}")
        print(f"Iterations: {num_iterations}")
        print(f"Validation samples per eval: {self.VAL_SAMPLES}")
        
        # Set random seed
        np.random.seed(initial_seed)
        torch.manual_seed(initial_seed)
        
        # Record how long it will take
        training_start = time.time()
        
        for iteration in range(num_iterations):
            iter_start = time.time()
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Generate random seeds for population
            seeds = np.random.randint(0, 2**30, size=self.POPULATION_SIZE, dtype=np.int64).tolist()
            rewards = []
            
            # Evaluating each perturbed model
            print("Evaluating population")
            for seed_idx, seed in enumerate(tqdm(seeds, desc="Population")):
                reward = self.evaluate_population_member(val_data, seed)
                rewards.append(reward)
                
                if (seed_idx + 1) % 10 == 0:
                    # Periodic cleanup for optimisation
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # ES update (aggregate perturbations)
            print("Applying ES update")
            self.es_update(seeds, rewards)
            
            # Save history
            rewards_array = np.array(rewards)
            self.history['mean_reward'].append(float(rewards_array.mean()))
            self.history['min_reward'].append(float(rewards_array.min()))
            self.history['max_reward'].append(float(rewards_array.max()))
            self.history['std_reward'].append(float(rewards_array.std()))
            
            iter_time = time.time() - iter_start
            
            # Print results
            print(f"Iteration {iteration + 1} Outputs:")
            print(f"  Mean Reward (ROUGE-1): {self.history['mean_reward'][-1]:.4f}")
            print(f"  Min Reward:            {self.history['min_reward'][-1]:.4f}")
            print(f"  Max Reward:            {self.history['max_reward'][-1]:.4f}")
            print(f"  Std Dev:               {self.history['std_reward'][-1]:.4f}")
            print(f"  Time: {iter_time:.2f}s")
            
            # Save checkpoint periodically
            if (iteration + 1) % save_every == 0:
                self.save_checkpoint(iteration + 1)
                self.save_history()
            
            # Cleanup
            gc.collect(); torch.cuda.empty_cache()
        
        total_time = time.time() - training_start
        

        print("Training finished")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        # show mean reward
        print(f"Final mean reward: {self.history['mean_reward'][-1]:.4f}")
        
        # Save final model
        print("Saving final model")
        final_path = os.path.join(self.output_directory, "final_model")
        self.model.save_pretrained(final_path)
        self.tokeniser.save_pretrained(final_path)
        print(f"Final model saved to: {final_path}")
        
        # Save history and plots
        self.save_history()
        self.plot_training_curves()
        
        return self.model
    
    def plot_training_curves(self):
        """Plot and save ES training curves for ESTrainer"""
        iterations = range(1, len(self.history['mean_reward']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Reward progression
        axes[0].plot(iterations, self.history['mean_reward'], 'b-o', label='Mean Reward', linewidth=2)
        axes[0].fill_between(
            iterations,
            np.array(self.history['mean_reward']) - np.array(self.history['std_reward']),
            np.array(self.history['mean_reward']) + np.array(self.history['std_reward']),
            alpha=0.3
        )
        axes[0].plot(iterations, self.history['max_reward'], 'g--', label='Max Reward', alpha=0.7)
        axes[0].plot(iterations, self.history['min_reward'], 'r--', label='Min Reward', alpha=0.7)
        axes[0].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('ROUGE-1 Score', fontsize=12, fontweight='bold')
        axes[0].set_title('ES Training Progress\n(FLAN-T5-base on BioLaySumm)', 
                         fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Reward variance
        axes[1].plot(iterations, self.history['std_reward'], 'm-o', linewidth=2)
        axes[1].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Reward Std Dev', fontsize=12, fontweight='bold')
        axes[1].set_title('Population Diversity\n(Exploration)', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_directory, "es_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")
        plt.show()
        plt.close()