# predict.py - An example of the trained model
# Author: Jevi Waugh
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from evaluate import load
from typing_extensions import Dict, Literal
import argparse
from dataset import BioDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class PredictSummary():
    
    def __init__(self, model_path, lora=False, model="google/flan-t5-base"):
        """This will intialise and set up the predictor from the saved model to show predictions
            and plots.

        Args:
            model_path (_type_): Path of the model
            lora (bool, optional): If Lora is being used. Defaults to False.
            model (str, optional): Model type for e.g. small or base. Defaults to "google/flan-t5-small".
        """
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokeniser = AutoTokenizer.from_pretrained(model if lora else model_path)
        
        if lora:
            # Load base model and LoRA adapter
            base = AutoModelForSeq2SeqLM.from_pretrained(model)
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            # Load full fine-tuned model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


        logger.info(f"Model has been loaded from {model_path} using {self.device}")
        self.model.eval()
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}")
        
    def _load_model(self, model: str) -> AutoModelForSeq2SeqLM:
        return AutoModelForSeq2SeqLM.from_pretrained(model)
    
    def predict_summary(self, text, max_length=256, num_beams=4, temperature=1.0):
        """This function will generate a layman summary for a radiology report.

        Args:
            text (_type_): Input radiology report
            max_length (int, optional): Maximum length of generated summary. Defaults to 256.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
        
        Returns:
            Generated layman text
            
        """
        # Prepare input with task prefix
        prefix = "Create a lay summary of this radiology report for a general audience:: "
        input_t = prefix + text
        
        # Tokenise
        inputs = self.tokeniser(input_t, max_length=512, truncation=True,
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
        summary = self.tokeniser.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def predict_batch_summaries(self, texts, max_length=256, num_beams=4, batch_size=8):
        """
        This will enerate summaries for multiple texts.
        
        Args:
            texts: List of input radiology reports
            max_length: Maximum length of generated summaries
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Prepare inputs
            prefix = "Create a lay summary of this radiology report for a general audience:: "
            input_texts = [prefix + text for text in batch_texts]
            
            inputs = self.tokeniser(input_texts, max_length=512,truncation=True, padding="max_length", return_tensors="pt").to(self.device)
            
            # Generate and decode
            with torch.no_grad():
                results = self.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],  
                                              max_length=max_length, num_beams=num_beams, early_stopping=True)
            batch_summaries = self.tokeniser.batch_decode(results, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        return summaries
    
def load_checkpoint_history(base_directory, num_epochs=3):
    """
    Load training history from a checkpoint JSON files.
    
    Args:
        base_dir: Base directory containing checkpoints folder
        num_epochs: Number of epochs to load (default: 3)
    
    Returns:
        Dictionary with history for all metrics
    """
    history = {
        'train_loss': [],
        'eval_loss': [],
        'rouge1': [],
        'rougeL': [],
        'rouge2': [],
        'rougeLsum': [],
        'gen_len': []
    }
    
    checkpoint_dir = os.path.join(base_directory, "checkpoints")
    
    for epoch in range(1, num_epochs + 1):
        checkpoint_p = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}", "checkpoint_info.json")
        
        try:
            with open(checkpoint_p, 'r') as f: checkpoint_results = json.load(f)
            
            history['train_loss'].append(checkpoint_results['train_loss'])
            history['eval_loss'].append(checkpoint_results['eval_loss'])
            history['rouge1'].append(checkpoint_results['rouge1'])
            history['rouge2'].append(checkpoint_results['rouge2'])
            history['rougeL'].append(checkpoint_results['rougeL'])
            history['rougeLsum'].append(checkpoint_results['rougeLsum'])
            history['gen_len'].append(checkpoint_results['gen_len'])
            
            print(f" Currently loading Epoch {epoch}")
        except FileNotFoundError:
            print(f"Checpoint not found for epoch {epoch}")
            return None
    
    return history

def plot_side_by_side_comparison(lora_history, fft_history, output_path):
    """
    Create 2x4 grid: 4 plots for LoRA, 4 plots for Full FT (3 epochs each).
    
    Args:
        lora_history: Dictionary with LoRA training history
        fft_history: Dictionary with Full Fine-Tuning training history
        output_path: Path to save the comparison plot
    """
    epochs = [1, 2, 3]
    
    # Create a figure with 2 rows (LoRA, FFT) and 4 columns (4 plot types)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    # TOP ROW: LoRA (FLAN-T5-base)
    # LoRA Plot 1: Loss curves
    axes[0, 0].plot(epochs, lora_history['train_loss'], 'b-o', label='Training Loss', linewidth=2.5, markersize=9)
    axes[0, 0].plot(epochs, lora_history['eval_loss'], 'r-o', label='Validation Loss', linewidth=2.5, markersize=9)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss\n(LoRA - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(epochs)
    
    # LoRA Plot 2: ROUGE scores
    axes[0, 1].plot(epochs, lora_history['rouge1'], 'g-o', label='ROUGE-1', linewidth=2.5, markersize=9)
    axes[0, 1].plot(epochs, lora_history['rouge2'], 'b-o', label='ROUGE-2', linewidth=2.5, markersize=9)
    axes[0, 1].plot(epochs, lora_history['rougeL'], 'r-o', label='ROUGE-L', linewidth=2.5, markersize=9)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('ROUGE Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('ROUGE Scores Over Training\n(LoRA - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(epochs)
    
    # LoRA Plot 3: Validation/Evaluation Loss 
    axes[0, 2].plot(epochs, lora_history['eval_loss'], 'r-o', label='Validation Loss', linewidth=2.5, markersize=9)
    axes[0, 2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Validation Loss (Detail)\n(LoRA - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xticks(epochs)
    
    # LoRA Plot 4: Generation length
    axes[0, 3].plot(epochs, lora_history['gen_len'], 'm-o', label='Avg Gen Length', linewidth=2.5, markersize=9)
    axes[0, 3].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 3].set_ylabel('Tokens', fontsize=12, fontweight='bold')
    axes[0, 3].set_title('Average Generation Length\n(LoRA - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[0, 3].legend(fontsize=10)
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].set_xticks(epochs)
    

    # BOTTOM ROW: Full Fine-Tuning (FLAN-T5-base)
    # FFT Plot 1: Loss curves
    axes[1, 0].plot(epochs, fft_history['train_loss'], 'b-o', label='Training Loss', linewidth=2.5, markersize=9)
    axes[1, 0].plot(epochs, fft_history['eval_loss'], 'r-o', label='Validation Loss', linewidth=2.5, markersize=9)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training and Validation Loss\n(Full FT - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(epochs)
    
    # FFT Plot 2: ROUGE scores
    axes[1, 1].plot(epochs, fft_history['rouge1'], 'g-o', label='ROUGE-1', linewidth=2.5, markersize=9)
    axes[1, 1].plot(epochs, fft_history['rouge2'], 'b-o', label='ROUGE-2', linewidth=2.5, markersize=9)
    axes[1, 1].plot(epochs, fft_history['rougeL'], 'r-o', label='ROUGE-L', linewidth=2.5, markersize=9)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('ROUGE Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('ROUGE Scores Over Training\n(Full FT - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(epochs)
    
    # FFT Plot 3: Validation/Evaluation  Loss zoomed
    axes[1, 2].plot(epochs, fft_history['eval_loss'], 'r-o', label='Validation Loss', linewidth=2.5, markersize=9)
    axes[1, 2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1, 2].set_title('Validation Loss (Detail)\n(Full FT - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(epochs)
    
    # FFT Plot 4: Generation length
    axes[1, 3].plot(epochs, fft_history['gen_len'], 'm-o', label='Avg Gen Length', linewidth=2.5, markersize=9)
    axes[1, 3].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 3].set_ylabel('Tokens', fontsize=12, fontweight='bold')
    axes[1, 3].set_title('Average Generation Length\n(Full FT - FLAN-T5-base)', fontsize=13, fontweight='bold', pad=12)
    axes[1, 3].legend(fontsize=10)
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_xticks(epochs)
    plt.suptitle('LoRA vs Full Fine-Tuning Comparison - First 3 Epochs\n(FLAN-T5-base, 247M parameters)',
                 fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"The Plot has been saved to: {output_path}")
    plt.close()

def printing_comparison(original_report, generated_summ, reference_summ):
    """This will print a formated comparison of the original report, generated and summary.

    Args:
        original_one (_type_): Original radiology report
        generated (_type_): Generated Layman summary
        reference (_type_): Reference layman summary
    """
    print("Original report: ")
    print(original_report)
    
    print("Generated Layman summary: ")
    print(generated_summ)
    
    print("Reference layman summary")
    print(reference_summ)
    
def inference(summary_predictor: PredictSummary, testing_input, number=4):
    """This will use the model and will show multiple predictions

    Args:
        summary_predictor (_type_): The predictor class
        testing_input (_type_): Testing input
        number (int, optional): Number of examples. Defaults to 4.
    """
    print(f"Inference on {number} examples")
    # Getting the actual length from one of the dictionary keys
    data_length = len(testing_input['radiology_report'])
    
    for i in range(min(number, data_length)):
        original_report = testing_input['radiology_report'][i]
        reference_summary = testing_input['layman_report'][i]
        print(f"Example {i + 1}: \n")
        # Generate summary
        gen_summary = summary_predictor.predict_summary(original_report)
        # show comparison
        printing_comparison(original_report, gen_summary, reference_summary)
        
def rouge_score(summary_predictor: PredictSummary, testing_data, num_samples=50):
    """Calculates rouge scores

    Args:
        summary_predictor (PredictSummary): Predictor class
        testing_data (_type_): Testing data (validation)
        num_samples (int, optional): Number of examples. Defaults to 50.

    Returns:
        _type_: None
    """
    rouge = load("rouge")
    try:
        reports = testing_data['radiology_report'][:num_samples]
    except Exception as error:
        logger.warning(f"Error: Could not load reports: {error}")
    try:   
        references = testing_data['layman_report'][:num_samples]
    except Exception as error:
        logger.warning(f"Error: Could not load references: {error}")
        
    # Take out the empty references
    valid_indices = [i for i, ref in enumerate(references) if ref and ref.strip()]
    
    if len(valid_indices) < num_samples:
        logger.warning(f"Warning: {len(valid_indices)} samples have non-empty reference summaries out of {num_samples}")
    
    if len(valid_indices) == 0:
        logger.warning("Critical Error: No valid reference summaries found."); return None
    
    print(f"{len(valid_indices)} samples for ROUGE calculation")
    
    # Using only valid samples
    reports = [reports[i] for i in valid_indices]
    ground_truth_reference = [references[i] for i in valid_indices]  
    
    preds = summary_predictor.predict_batch_summaries(reports)
    
    # Use Porter stemmer to strip word suffixes
    outputs = rouge.compute(predictions=preds, references=ground_truth_reference, use_stemmer=True)
    
    # print all metrics
    _print_rouge_metrics(outputs)
    
def _print_rouge_metrics(metrics: Dict[str, float], 
                        metric: Literal["rouge1", "rouge2", "rougeL", "rougeLsum", "all"]="rouge1") -> None:
    
    keys = metric.keys() if metric["all"] else [metric]
    for k in keys:
        print(f"{k.upper()}: {metrics[k]:.4f}")
    
def train_history_diff(lora_base_directory, fft_base_directory, output_path, num_epochs=3, chosen_epoch_diff=2, model_name="FLAN-T5-base"):
    """Creates comparison plots through loaded checkpoint histories. This will also call plot_side_by_side_comparison().

    Args:
        lora_base_dir (_type_): Directory to LoRA checkpoints
        fft_base_dir (_type_): Directory to Full-fine tuning
        output_path (_type_): Path to save the comparison plot
        num_epochs (int, optional): Number of epochs. Defaults to 3.
        chosen_epoch_diff (int): Chosen epoch to view the diff (index starts at 0)
    """
    
    try:
        loRA_history = load_checkpoint_history(lora_base_directory, num_epochs=num_epochs)
    except Exception as error:
        logger.warning("Could not load checkpoint for LoRA: {error}")
    
    try:
        Fft_history = load_checkpoint_history(fft_base_directory, num_epochs=num_epochs)
    except Exception as error:
        logger.warning("Could not load checkpoint for full fine tuning: {error}")
        
    plot_side_by_side_comparison(loRA_history, Fft_history, output_path)
    
    print(f"LoRA for {model_name} - Epoch {num_epochs}:")
    print(f"Training Loss:   {loRA_history['train_loss'][chosen_epoch_diff]:.4f}")
    print(f"Validation Loss: {loRA_history['eval_loss'][chosen_epoch_diff]:.4f}")
    print(f"ROUGE-L:         {loRA_history['rougeL'][chosen_epoch_diff]:.4f}")
    print(f"ROUGE-1:         {loRA_history['rouge1'][chosen_epoch_diff]:.4f}")
    print(f"ROUGE-2:         {loRA_history['rouge2'][chosen_epoch_diff]:.4f}")
    print(f"Gen Length:      {loRA_history['gen_len'][chosen_epoch_diff]:.2f}")
    print("\n")
    print(f"Full Fine-Tuning (FLAN-T5-base) - Epoch {num_epochs}:")
    print(f"Training Loss:   {Fft_history['train_loss'][chosen_epoch_diff]:.4f}")
    print(f"Validation Loss: {Fft_history['eval_loss'][chosen_epoch_diff]:.4f}")
    print(f"ROUGE-L:         {Fft_history['rougeL'][chosen_epoch_diff]:.4f}")
    print(f"ROUGE-1:         {Fft_history['rouge1'][chosen_epoch_diff]:.4f}")
    print(f"ROUGE-2:         {Fft_history['rouge2'][chosen_epoch_diff]:.4f}")
    print(f"Gen Length:      {Fft_history['gen_len'][chosen_epoch_diff]:.2f}")
    print("\n")
    print("Difference (LoRA - Full FT):")
    print(f"Training Loss:   {loRA_history['train_loss'][chosen_epoch_diff]-Fft_history['train_loss'][chosen_epoch_diff]:+.4f}")
    print(f"Validation Loss: {loRA_history['eval_loss'][chosen_epoch_diff]-Fft_history['eval_loss'][chosen_epoch_diff]:+.4f}")
    print(f"ROUGE-L:         {loRA_history['rougeL'][chosen_epoch_diff]-Fft_history['rougeL'][chosen_epoch_diff]:+.4f}")
    print(f"ROUGE-1:         {loRA_history['rouge1'][chosen_epoch_diff]-Fft_history['rouge1'][chosen_epoch_diff]:+.4f}")
    print(f"ROUGE-2:         {loRA_history['rouge2'][chosen_epoch_diff]-Fft_history['rouge2'][chosen_epoch_diff]:+.4f}")
    print("\n")
    print(f"The path to the saved history for LoRA is: {lora_base_directory}")
    print(f"The path to the saved history for Full-Fine Tuning is: {fft_base_directory}")
    
def main():
    """Main script with command line arguments."""
    
    parser = argparse.ArgumentParser(description='Test trained summarisation model.')
    # Model path
    parser.add_argument('--model_path', type=str,
                       help='The path to trained model for predictions')
    # ROUGE
    parser.add_argument('--calculate_rouge', action='store_true',
                       help='Calculate ROUGE scores on test dataset')
    # LoRA
    parser.add_argument('--is_lora', action='store_true',
                       help='If model is a LoRA adapter')
    # Comparison
    parser.add_argument('--compare_training', action='store_true',
                       help='Compare Full Fine-Tuning and  LoRA training histories')
    # LoRA Dir
    parser.add_argument('--lora_dir', type=str,
                       help='Base directory for LoRA checkpoints')
    # FFT Dir
    parser.add_argument('--fft_dir', type=str,
                       help='Base directory for Full Fine-Tuning checkpoints')
    # Output Path
    parser.add_argument('--output_plot', type=str, default='./outputs/comparison_plot.png',
                       help='Path to save comparison plot')
    # EPOCHS
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs to analyse')
    # EPOCH EXAMPLES
    parser.add_argument('--num_examples', type=int, default=3,
                       help='Number of examples to show')
    
    args = parser.parse_args()
    
    # Compare training histories if requested
    if args.compare_training:
        if not args.lora_dir or not args.fft_dir:
            print("Error: --lora_dir and --fft_dir are required to compare for training")
            return
        
        train_history_diff(lora_base_dir=args.lora_dir,  fft_base_dir=args.fft_dir, output_path=args.output_plot, num_epochs=args.num_epochs)
    
    # Run predictions if the model_path has been provided
    if args.model_path:
        # Load test data
        print("Loading test dataset")
        dataloader = BioDatasetLoader()
        dataloader.load_dataset(splits=("test",))
        
        # We use validation so that we can compare with the reference.
        val_data = dataloader.validation
        
        print(f"Test dataset size: {len(val_data)}")
        
        # Load predictor
        predictor = PredictSummary(
            model_path=args.model_path,
            lora=args.is_lora
        )
        print("="*10)
        # Demonstrate predictions
        inference(predictor, val_data, num_examples=args.num_examples)
        
        # Calculate ROUGE if needed
        if args.calculate_rouge:
            rouge_score(predictor, val_data, num_samples=100)
        
        print("Demonstration for predictions have been shown.")

if __name__ == "__main__":
    main()