# predict.py - An example of the trained model
# Author: Jevi Waugh


class PredictSummary():
    
    def __init__(self, model_path, lora=False, model="google/flan-t5-small"):
        """This will intialise and set up the predictor from the saved model to show predictions
            and plots.

        Args:
            model_path (_type_): Path of the model
            lora (bool, optional): If Lora is being used. Defaults to False.
            model (str, optional): Model type for e.g. small or base. Defaults to "google/flan-t5-small".
        """
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokemiser = AutoTokenizer.from_pretrained(model_path)
        from peft import PeftModel
        # Load lora byt loading the autoclass model and its correspondinf file path
        if lora: self.model =  PeftModel.from_pretraind(AutoModelForSeq2SeqLM.from_pretrained(model), model_path)
        # Otherwise just load the model withtout lora
        else: self.model = AutoModelForSeq2SeqLM.from_pretrained(model)