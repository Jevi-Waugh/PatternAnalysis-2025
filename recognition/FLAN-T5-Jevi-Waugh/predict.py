# predict.py - An example of the trained model
# Author: Jevi Waugh
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel
import logging


class PredictSummary():
    
    def __init__(self, model_path, lora=False, model="google/flan-t5-small"):
        """This will intialise and set up the predictor from the saved model to show predictions
            and plots.

        Args:
            model_path (_type_): Path of the model
            lora (bool, optional): If Lora is being used. Defaults to False.
            model (str, optional): Model type for e.g. small or base. Defaults to "google/flan-t5-small".
        """
        logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokemiser = AutoTokenizer.from_pretrained(model_path)
        # Load lora byt loading the autoclass model and its correspondinf file path
        if lora: self.model =  PeftModel.from_pretraind(self._load_model(model), model_path)
        # Otherwise just load the model withtout lora
        else: self.model = self._load_model(model)
        
        # Transfer model to computer for compute
        self.model.to(self.device)
        logger.info(f"Model has been loaded from {model_path} using {self.device}")
        
        
        
    def _load_model(self, model) -> AutoModelForSeq2SeqLM:
        return AutoModelForSeq2SeqLM.from_pretrained(model)
    
    
    def plot() -> None:
        pass
    
    def predict_summary() -> None:
        pass
    