"""
Model Evaluation Script for AG News Classification

This module handles:
- Model evaluation on test set
- Comprehensive metrics calculation
- Confusion matrix visualization
- Per-class performance analysis
"""

import logging
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AGNewsEvaluator:
    """Evaluator for AG News classification model."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize evaluator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.class_names = self.config['class_labels']
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> torch.device:
        """Setup computation device (GPU/CPU)."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
        """
        logger.info(f"Loading model from {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def load_dataset(self, dataset_path: str = "data/processed") -> None:
        """
        Load dataset.
        
        Args:
            dataset_path: Path to preprocessed dataset
        """
        logger.info(f"Loading dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        logger.info("Dataset loaded successfully")
    
    def evaluate(self, split: str = 'test', batch_size: int = 32) -> Dict:
        """
        Evaluate model on specified split.
        
        Args:
            split: Dataset split ('test', 'validation', or 'train')
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")
        
        logger.info(f"Evaluating on {split} set...")
        
        # Create data loader
        data_loader = DataLoader(
            self.dataset[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['optimization']['num_workers'],
            pin_memory=self.config['optimization']['pin_memory'],
        )
        
        all_preds = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=self.config['evaluation']['average_type']
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'split': split,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / len(data_loader),
            'predictions': all_preds,
            'labels': all_labels,
            'logits': all_logits,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict) -> None:
        """
        Print evaluation metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics
        """
        logger.info(f"\n=== Evaluation Results ({metrics['split']} set) ===")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"Loss:      {metrics['loss']:.4f}")
        
        logger.info("\nPer-Class Metrics:")
        for class_id, class_name in self.class_names.items():
            logger.info(
                f"  {class_name:12s} - "
                f"Precision: {metrics['precision_per_class'][class_id]:.4f} | "
                f"Recall: {metrics['recall_per_class'][class_id]:.4f} | "
                f"F1: {metrics['f1_per_class'][class_id]:.4f}"
            )
    
    def plot_confusion_matrix(self, metrics: Dict, output_path: str = "logs/confusion_matrix.png") -> None:
        """
        Plot confusion matrix.
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_path: Path to save plot
        """
        cm = metrics['confusion_matrix']
        class_names_list = [self.class_names[i] for i in range(len(self.class_names))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names_list,
            yticklabels=class_names_list,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f"Confusion Matrix - {metrics['split']} Set")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
        plt.close()
    
    def plot_per_class_metrics(self, metrics: Dict, output_path: str = "logs/per_class_metrics.png") -> None:
        """
        Plot per-class metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_path: Path to save plot
        """
        class_names_list = [self.class_names[i] for i in range(len(self.class_names))]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Precision
        axes[0].bar(class_names_list, metrics['precision_per_class'], color='steelblue')
        axes[0].set_title('Precision per Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1].bar(class_names_list, metrics['recall_per_class'], color='coral')
        axes[1].set_title('Recall per Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[2].bar(class_names_list, metrics['f1_per_class'], color='lightgreen')
        axes[2].set_title('F1-Score per Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_ylim([0, 1])
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to {output_path}")
        plt.close()
    
    def plot_prediction_confidence(self, metrics: Dict, output_path: str = "logs/confidence_distribution.png") -> None:
        """
        Plot prediction confidence distribution.
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_path: Path to save plot
        """
        # Get max probability for each prediction
        probabilities = torch.softmax(torch.tensor(metrics['logits']), dim=1).numpy()
        max_probs = probabilities.max(axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = metrics['predictions'] == metrics['labels']
        correct_probs = max_probs[correct_mask]
        incorrect_probs = max_probs[~correct_mask]
        
        plt.figure(figsize=(10, 6))
        plt.hist(correct_probs, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_probs, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confidence distribution plot saved to {output_path}")
        plt.close()


def main():
    """Main function for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AG News Model Evaluation")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--model-path', default='models/best_model.pt', help='Model path')
    parser.add_argument('--dataset-path', default='data/processed', help='Dataset path')
    parser.add_argument('--split', default='test', help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    evaluator = AGNewsEvaluator(args.config)
    evaluator.load_model(args.model_path)
    evaluator.load_dataset(args.dataset_path)
    
    metrics = evaluator.evaluate(split=args.split, batch_size=args.batch_size)
    evaluator.print_metrics(metrics)
    
    if args.visualize:
        evaluator.plot_confusion_matrix(metrics)
        evaluator.plot_per_class_metrics(metrics)
        evaluator.plot_prediction_confidence(metrics)


if __name__ == "__main__":
    main()
