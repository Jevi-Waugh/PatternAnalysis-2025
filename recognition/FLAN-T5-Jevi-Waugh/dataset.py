# dataset.py
# Author: Jevi Waugh
import logging
from typing_extensions import Literal, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from transformers import DataCollatorForSeq2Seq
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

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
        return self.tokeniser
    
    def preprocessing(self, data):
        """Preproccesses data by prepending summarisation query to input and further
           tokenises the data.

        Args:
            data (_type_): The data being processed. For example a radiology report.

        Returns:
            _type_: The inputs of the modle
        """
        prefix = "summarise: "
        prefix2 = "Create a lay summary of this radiology report for a general audience:: "
        
        # ask for shakes if i can do system prompts/
        # system_prompt = ""
        # The model needs to know that this is a summarisation task
        summ_input = [prefix2 + d for d in data["radiology_report"]]
        self._load_tokeniser()
        # max length by 128
        MODEL_INPUTS = self.tokeniser(summ_input, max_length=512, truncation=True, padding="max_length")
        LABELS = self.tokeniser(text_target=data["layman_report"], max_length=256, truncation=True, padding="max_length")
        MODEL_INPUTS["labels"] = LABELS["input_ids"]
        return MODEL_INPUTS
    

    def process_dataset(self, dataset):
        """Preprocesses the dataset 

        Args:
            dataset (_type_): Dataset

        Returns:
            _type_: Proccessed dataset
        """
        # perhaps remove columns??
        return dataset.map(self.preprocessing, batched=True)
    
    def _padding(self): 
        if not hasattr(self, "data_collator"):
            self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokeniser, model=self.model_name)
            print("Data collator initialized")
        

def main(): 
    # Testing dataloader
    BioLoader = BioDatasetLoader(); 
    # we can choose which ones we wanna load - 
    BioLoader.load_dataset()
    print(f"Training size: {BioLoader.get_size(BioLoader.train)}")
    print(f"Testing size: {BioLoader.get_size(BioLoader.test)}")
    print(f"Validation size: {BioLoader.get_size(BioLoader.validation)}")
    print(f"Dataset size: {len(BioLoader)}")
    
    # Print first data to evaluate 
    print(BioLoader.train[0])
    
    # Load tokeniser
    BioLoader._load_tokeniser()
    
    # pad inputs
    BioLoader._padding()
    
if __name__ == "__main__": main()