# Fine tuning procedure is in this file
from transformers import AutoTokenizer
import torch.nn as nn
from dataset import BioDatasetLoader
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq

class FLAN_T5:
    def __init__(self, DataLoader, model_name="google/flan-t5-small"):
        self.dataloader = DataLoader()
        self.dataloader.load_dataset()
        self.tokeniser = self.dataloader._load_tokeniser(model_name)
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.data_collator = None

    def _padding(self):
        """Ensure the data collator exists."""
        if not self.data_collator:
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokeniser,
                model=self.model
            )

    def load_model(self):
        self._padding()
        training_args = Seq2SeqTrainingArguments(
            output_dir="BioLaySumm",
            eval_strategy="epoch",
            learning_rate=5e-4,           # increased learning rate for LoRA
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            bf16=True,
            fp16=False,
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataloader.process_dataset(self.dataloader.train),
            eval_dataset=self.dataloader.process_dataset(self.dataloader.validation),
            tokenizer=self.tokeniser,
            data_collator=self.data_collator,
            compute_metrics=self.dataloader.compute_rouge_scores,
        )
        return trainer


class FLAN_T5_loRA(FLAN_T5):
    """This class will be used to finetune FLAN-T5 via LoRA

    Args:
        FLAN_T5 (_type_): _description_
    """
    def __init__(self, DataLoader, model_name="google/flan-t5-small"):
        # Call parent constructor
        super().__init__(DataLoader, model_name)
        # Apply LoRA configuration immediately after initialization
        self.loRA_Config()
    
    def loRA_Config(self):
        """Apply LoRA configuration to the model."""
        lora_config = LoraConfig(
            r=16,    # increased rank for more capacity
            lora_alpha=32,  # increased scaling factor
            target_modules=["q", "v", "k", "o", "wi", "wo"],  # target more layers
            lora_dropout=0.1,   # slightly increased dropout
            bias="none",
            task_type="SEQ_2_SEQ_LM" # important for T5
        )

        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied. Trainable parameters: {self.model.print_trainable_parameters()}")
        return self.model


class FLAN_T5_FULLFINETUNING(FLAN_T5):
    """This class will be used to Fine Tune Flan-T5

    Args:
        FLAN_T5 (_type_): _description_
    """
    pass


class FLAN_T5_SPECIAL(FLAN_T5):
    """This is a special strategy that I will reveal once it works.

    Args:
        FLAN_T5 (_type_): _description_
    """
    pass


def main():
    model = FLAN_T5_loRA(BioDatasetLoader, model="google/flan-t5-base")
    trainer = model.dataloader.load_model()
    trainer.train()
    trainer.save_model("flan_t5_biolaysumm")
    trainer.tokenizer.save_pretrained("flan_t5_biolaysumm")
    
if __name__ == "__main__": 
    main()