from datasets import load_dataset
import logging
from typing_extensions import Literal
# Author: Jevi Waugh

class BioDatasetLoader():
    """This dataset Loader loads the BioLaySumm dataset and loads the required 
    """
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, link=None, workers=0):
        if not link: 
            self.HUGGING_FACE_DATASET = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
        self.train = None
        self.test = None
        self.validation = None
        self.num_workers = workers
        self.datatype = ["train", "test", "validation"]

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
        
    def __len__(self) -> int:
        """Returns cumulative dataset size.

        Returns:
            _type_: Size of the entire dataset.
        """
        return sum(len(data) for data in [self.train, self.test, self.validation] if data is not None)
    
    def get_size(self, dataset: Literal["train", "test", "validation"]) -> int:
        """returns the size of a specific dataset.

        Args:
            dataset (Literal[&quot;train&quot;, &quot;test&quot;, &quot;validation&quot;]): dataset. for e.g. self.train

        Returns:
            _type_: Size of the given dataset
        """
        return len(dataset)
    
    def __repr__(self) -> None:
        """A summary of the dataset loader.

        Returns:
            _type_: None
        """
        return (f"num_workers={self.num_workers},"  
                f"Length of the dataset {len(self)}")
