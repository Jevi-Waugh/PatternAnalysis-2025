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
        self.model.eval()
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}")
        
        
    def _load_model(self, model) -> AutoModelForSeq2SeqLM:
        return AutoModelForSeq2SeqLM.from_pretrained(model)
    
    
    def predict_summary(self, text, max_length=256, num_beams=4, temperature=1.0):
        """This function will generate a layman summary for a radiology report.

        Args:
            text (_type_): Input radiology report
            max_length (int, optional): Maximum length of generated summary. Defaults to 256.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
        
        Returns:
            Generated summary text
            
        """
        # Prepare input with task prefix
        prefix = "Create a lay summary of this radiology report for a general audience:: "
        input_t = prefix + text
        
        # Tokenise
        inputs = self.tokenizer(input_t, max_length=512, truncation=True,
                                padding="max_length", return_tensors="pt").to(self.device)
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    
    
    
    
    
    
def load_checkpoint():
    pass


def plot() -> None:
    pass