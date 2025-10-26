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
    def __init__(self, DataLoader, model_name="google/flan-t5-small"):
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



class FLAN_T5_SPECIAL(FLAN_T5):
    """This is a special strategy that I will reveal once it works.

    Args:
        FLAN_T5 (_type_): _description_
    """
    pass


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


def main():
    model = FLAN_T5_loRA(BioDatasetLoader, model="google/flan-t5-base")
    trainer = model.dataloader.load_model()
    trainer.train()
    trainer.save_model("flan_t5_biolaysumm")
    trainer.tokenizer.save_pretrained("flan_t5_biolaysumm")
    
if __name__ == "__main__": 
    main()