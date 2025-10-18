import logging
from typing_extensions import Literal, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from transformers import DataCollatorForSeq2Seq
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
# Author: Jevi Waugh

class BioDatasetLoader():
    """This dataset Loader loads the BioLaySumm dataset and loads the required data.
    """
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, link=None, workers=0):
        if not link: 
            self.HUGGING_FACE_DATASET = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
        self.train = None
        self.test = None
        self.validation = None
        self.num_workers = workers
        self.tokeniser = None
        self.model_name = None

    def __len__(self) -> int:
        """Returns cumulative dataset size.

        Returns:
            _type_: Size of the entire dataset.
        """
        return sum(len(data) for data in [self.train, self.test, self.validation] if data is not None)
    
    def __repr__(self) -> None:
        """A summary of the dataset loader.

        Returns:
            _type_: None
        """
        return (f"num_workers={self.num_workers},"  
                f"Length of the dataset {len(self)}")
        
    def get_size(self, dataset: Literal["train", "test", "validation"]) -> int:
        """returns the size of a specific dataset.

        Args:
            dataset (Literal[&quot;train&quot;, &quot;test&quot;, &quot;validation&quot;]): dataset. for e.g. self.train

        Returns:
            _type_: Size of the given dataset
        """
        return len(dataset)
    def load_dataset(self, splits=("train", "test", "validation")) -> None:
        """This loads the dataset and stores in the datasetloader. If splits is empty, then the loader
            funtion will load all train, test and validation datasets if available.

        Args:
            splits (tuple, optional): _description_. Defaults to ("train", "test", "validation").
        """
        data = {}
        for split in splits:
            try:
                data[split] = load_dataset(self.HUGGING_FACE_DATASET, split=split)
                self.logger.info("Successfully loaded {split} dataset.")
            except Exception as error:
                self.logger.warning(f"Warning: could not load {split}: {error}")
                
        self.train = data.get("train")
        self.test = data.get("test")
        self.validation = data.get("validation")
        
    
    def _load_tokeniser(self, checkpoint=None):
        # use small model
        if checkpoint is None: checkpoint = "google/flan-t5-small"
        self.model_name = checkpoint
        self.tokeniser = AutoTokenizer.from_pretrained(checkpoint)
    
    def preprocessing(self, data):
        """Preproccesses data by prepending summarisation query to input and further
           tokenises the data.

        Args:
            data (_type_): The data being processed. For example a radiology report.

        Returns:
            _type_: The inputs of the modle
        """
        prefix = "summarise: "
        
        # ask for shakes if i can do system prompts/
        # system_prompt = ""
        # The model needs to know that this is a summarisation task
        summ_input = [prefix + d for d in data["radiology_report"]]
        self._load_tokeniser()
        # max length by 128
        MODEL_INPUTS = self.tokeniser(summ_input, max_length=128, truncation=True)
        LABELS = self.tokeniser(text_target=data["layman_report"], max_length=128, truncation=True)
        MODEL_INPUTS["labels"] = LABELS["input_ids"]
        return MODEL_INPUTS
    

    def process_dataset(self, dataset):
        """Preprocesses the dataset 

        Args:
            dataset (_type_): Dataset

        Returns:
            _type_: Proccessed dataset
        """
        return dataset.map(self.preprocessing, batched=True)
    
    def _paddding(self): 
        if not hasattr(self, "data_collator"):
            self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokeniser, model=self.model_name)
            

    def compute_rouge_scores(self, eval_pred: Tuple):
        """This computes all rouge scores (rouge1, rouge2, rougeL, rougeLsum).

        Args:
            eval_pred (_type_): Tuple passed by the trainer durig an evaluation which has token
                                IDs output bu the model. The second part of the tuple are the labels
                                are the token IDs of the reference sumamries.

        Returns:
            _type_: _description_
        """
        rouge = evaluate.load("rouge")
        predictions, labels = eval_pred
        # Transforming the predictions to human readable texts, remove cls/eos or other padding
        decoded_preds = self.tokeniser.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokeniser.pad_token_id)
        # Decode label IDs into reference text strings.
        decoded_labels = self.tokeniser.batch_decode(labels, skip_special_tokens=True)
        # Use stemmer to normlaise certain words.
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokeniser.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {key: round(val, 4) for key, val in result.items()}
    
    def load_model(self) -> Seq2SeqTrainer:
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        training_args = Seq2SeqTrainingArguments(
            output_dir="BioLaySumm",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True, #change to bf16=True for XPU
            push_to_hub=False,
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.process_dataset(self.train),
            eval_dataset=self.process_dataset(self.validation),
            processing_class=self.tokeniser,
            data_collator=self.data_collator,
            compute_metrics=self.compute_rouge_scores,
        )
        return trainer
        

def main(): 
    # Testing dataloader
    BioLoader = BioDatasetLoader(); 
    # we can choose which ones we wanna load - 
    BioLoader.load_dataset()
    print(f"Training size: {BioLoader.get_size(BioLoader.train)}")
    print(f"Testing size: {BioLoader.get_size(BioLoader.test)}")
    print(f"Validation size: {BioLoader.get_size(BioLoader.validation)}")
    print(f"Dataset size: {len(BioLoader)}")
    
    # print(dataset.train[0])
    # Load tokeniser
    BioLoader._load_tokeniser()
    
    # pad inputs
    BioLoader._paddding()
    
    # Load model
    trainer = BioLoader.load_model()
    
    # # train
    # trainer.train()
    
    
if __name__ == "__main__": main()