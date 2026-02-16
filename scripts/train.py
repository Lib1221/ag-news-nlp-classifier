"""
Model Training Script for AG News Classification

This module handles:
- Model initialization and fine-tuning
- Training loop with validation
- Hyperparameter optimization using Optuna
- Checkpoint saving and early stopping
- Weights & Biases integration
"""

import os
import logging
import yaml
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna
from optuna.trial import Trial

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AGNewsTrainer:
    """Trainer for AG News classification model."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Setup logging
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B if enabled
        if self.config['wandb']['enabled'] and WANDB_AVAILABLE:
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                config=self.config,
                tags=self.config['wandb']['tags'],
                notes=self.config['wandb']['notes']
            )
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> torch.device:
        """Setup computation device (GPU/CPU)."""
        device_type = self.config['device']['type']
        if device_type == 'cuda' and torch.cuda.is_available():
            device_id = self.config['device']['device_id']
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        elif device_type == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Metal Performance Shaders (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def load_dataset(self, dataset_path: str = "data/processed") -> DatasetDict:
        """
        Load preprocessed dataset.
        
        Args:
            dataset_path: Path to preprocessed dataset
            
        Returns:
            Loaded DatasetDict
        """
        logger.info(f"Loading dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        logger.info("Dataset loaded successfully")
        return self.dataset
    
    def initialize_model(self) -> None:
        """Initialize model and tokenizer."""
        model_name = self.config['model']['name']
        num_classes = self.config['model']['num_classes']
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=self.config['model']['hidden_dropout_prob'],
            attention_probs_dropout_prob=self.config['model']['attention_probs_dropout_prob'],
        )
        self.model.to(self.device)
        logger.info("Model loaded successfully")
    
    def create_data_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_loader = DataLoader(
            self.dataset['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['optimization']['num_workers'],
            pin_memory=self.config['optimization']['pin_memory'],
        )
        
        val_loader = DataLoader(
            self.dataset['validation'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['optimization']['num_workers'],
            pin_memory=self.config['optimization']['pin_memory'],
        )
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(
        self,
        train_loader: DataLoader,
        learning_rate: float,
        epochs: int,
        warmup_steps: int,
    ) -> Tuple:
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            train_loader: Training data loader
            learning_rate: Learning rate
            epochs: Number of epochs
            warmup_steps: Number of warmup steps
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=self.config['training']['adam_epsilon'],
            weight_decay=self.config['training']['weight_decay'],
        )
        
        total_steps = len(train_loader) * epochs
        
        if self.config['scheduler']['type'] == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config['scheduler']['type'] == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.config['scheduler']['num_cycles'],
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Log progress
            if (step + 1) % self.config['logging']['log_interval'] == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(
                    f"Epoch {epoch + 1} | Step {step + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                
                if WANDB_AVAILABLE and self.config['wandb']['enabled']:
                    wandb.log({
                        'train_loss': avg_loss,
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch': epoch,
                    })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
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
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
        }
        
        logger.info(
            f"Epoch {epoch + 1} | Validation - "
            f"Loss: {metrics['val_loss']:.4f} | "
            f"Accuracy: {metrics['val_accuracy']:.4f} | "
            f"F1: {metrics['val_f1']:.4f}"
        )
        
        if WANDB_AVAILABLE and self.config['wandb']['enabled']:
            wandb.log({**metrics, 'epoch': epoch})
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
        """
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_best_model(self) -> None:
        """Save best model."""
        model_dir = Path(self.config['checkpoint']['save_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "best_model.pt"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Best model saved to {model_path}")
    
    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        warmup_steps: Optional[int] = None,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs (uses config if None)
            batch_size: Batch size (uses config if None)
            learning_rate: Learning rate (uses config if None)
            warmup_steps: Warmup steps (uses config if None)
            
        Returns:
            Dictionary with training results
        """
        # Use config values if not provided
        epochs = epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['training']['batch_size']
        learning_rate = learning_rate or self.config['training']['learning_rate']
        warmup_steps = warmup_steps or self.config['training']['warmup_steps']
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Load dataset if not already loaded
        if self.dataset is None:
            self.load_dataset()
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(batch_size)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(
            train_loader, learning_rate, epochs, warmup_steps
        )
        
        logger.info("Starting training...")
        logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
        }
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            training_history['train_loss'].append(train_loss)
            
            # Validate
            metrics = self.validate(val_loader, epoch)
            training_history['val_loss'].append(metrics['val_loss'])
            training_history['val_accuracy'].append(metrics['val_accuracy'])
            training_history['val_f1'].append(metrics['val_f1'])
            
            # Save checkpoint
            if self.config['checkpoint']['save_best_only']:
                if metrics['val_accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = metrics['val_accuracy']
                    self.save_checkpoint(epoch, metrics)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                    # Early stopping
                    if (self.config['early_stopping']['enabled'] and
                        self.patience_counter >= self.config['early_stopping']['patience']):
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        # Save best model
        self.save_best_model()
        
        logger.info("Training completed!")
        return training_history


def main():
    """Main function for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AG News Model Training")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, help='Warmup steps')
    parser.add_argument('--dataset-path', default='data/processed', help='Dataset path')
    
    args = parser.parse_args()
    
    trainer = AGNewsTrainer(args.config)
    trainer.load_dataset(args.dataset_path)
    trainer.initialize_model()
    
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
