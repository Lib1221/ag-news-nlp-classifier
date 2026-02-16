"""
Data Loading and Preprocessing Module for AG News Classification

This module handles:
- Downloading AG News dataset
- Preprocessing and tokenization
- Creating train/validation/test splits
- Data exploration and visualization
"""

import os
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AGNewsDataLoader:
    """Data loader and preprocessor for AG News dataset."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data loader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.dataset: Optional[DatasetDict] = None
        self.class_names = self.config['class_labels']
        
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def download_dataset(self) -> DatasetDict:
        """
        Download AG News dataset from Hugging Face.
        
        Returns:
            DatasetDict with train and test splits
        """
        logger.info("Downloading AG News dataset...")
        try:
            dataset = load_dataset('ag_news')
            self.dataset = dataset
            logger.info(f"Dataset downloaded successfully")
            logger.info(f"Train samples: {len(dataset['train'])}")
            logger.info(f"Test samples: {len(dataset['test'])}")
            return dataset
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def initialize_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """
        Initialize tokenizer from pre-trained model.
        
        Args:
            model_name: Name of pre-trained model
            
        Returns:
            PreTrainedTokenizer instance
        """
        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
        return self.tokenizer
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize text examples.
        
        Args:
            examples: Dictionary with 'text' and 'label' keys
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_length']
        )
    
    def preprocess_dataset(self, model_name: str) -> DatasetDict:
        """
        Preprocess dataset with tokenization and train/val/test split.
        
        Args:
            model_name: Name of pre-trained model
            
        Returns:
            Preprocessed DatasetDict
        """
        if self.dataset is None:
            self.download_dataset()
        
        self.initialize_tokenizer(model_name)
        
        logger.info("Tokenizing dataset...")
        
        # Tokenize all splits
        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        
        # Create train/validation split from original train set
        logger.info("Creating train/validation/test splits...")
        train_split = tokenized_dataset['train'].train_test_split(
            test_size=self.config['data']['val_split'] / (1 - self.config['data']['test_split']),
            seed=self.config['data']['random_seed']
        )
        
        # Rename splits
        processed_dataset = DatasetDict({
            'train': train_split['train'],
            'validation': train_split['test'],
            'test': tokenized_dataset['test']
        })
        
        logger.info(f"Train samples: {len(processed_dataset['train'])}")
        logger.info(f"Validation samples: {len(processed_dataset['validation'])}")
        logger.info(f"Test samples: {len(processed_dataset['test'])}")
        
        self.dataset = processed_dataset
        return processed_dataset
    
    def explore_dataset(self) -> None:
        """Explore and visualize dataset characteristics."""
        if self.dataset is None:
            self.download_dataset()
        
        logger.info("Exploring dataset...")
        
        # Get class distribution
        train_labels = self.dataset['train']['label']
        test_labels = self.dataset['test']['label']
        
        # Check if validation split exists
        if 'validation' in self.dataset:
            val_labels = self.dataset['validation']['label']
            has_val = True
        else:
            val_labels = []
            has_val = False
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Train distribution
        pd.Series(train_labels).value_counts().sort_index().plot(
            kind='bar', ax=axes[0], color='steelblue'
        )
        axes[0].set_title('Train Set Class Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels([self.class_names[i] for i in range(4)], rotation=45)
        
        # Validation distribution
        pd.Series(val_labels).value_counts().sort_index().plot(
            kind='bar', ax=axes[1], color='coral'
        )
        axes[1].set_title('Validation Set Class Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_xticklabels([self.class_names[i] for i in range(4)], rotation=45)
        
        # Test distribution
        pd.Series(test_labels).value_counts().sort_index().plot(
            kind='bar', ax=axes[2], color='lightgreen'
        )
        axes[2].set_title('Test Set Class Distribution')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('Count')
        axes[2].set_xticklabels([self.class_names[i] for i in range(4)], rotation=45)
        
        plt.tight_layout()
        plt.savefig('logs/class_distribution.png', dpi=150, bbox_inches='tight')
        logger.info("Class distribution plot saved to logs/class_distribution.png")
        
        # Print statistics
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Total samples: {len(train_labels) + len(val_labels) + len(test_labels)}")
        logger.info(f"Train: {len(train_labels)} ({len(train_labels)/(len(train_labels) + len(val_labels) + len(test_labels))*100:.1f}%)")
        logger.info(f"Validation: {len(val_labels)} ({len(val_labels)/(len(train_labels) + len(val_labels) + len(test_labels))*100:.1f}%)")
        logger.info(f"Test: {len(test_labels)} ({len(test_labels)/(len(train_labels) + len(val_labels) + len(test_labels))*100:.1f}%)")
        
        logger.info("\nClass Distribution:")
        for class_id, class_name in self.class_names.items():
            count = (np.array(train_labels) == class_id).sum()
            logger.info(f"  {class_name}: {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    def save_dataset(self, output_dir: str = "data/processed") -> None:
        """
        Save processed dataset to disk.
        
        Args:
            output_dir: Directory to save dataset
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call preprocess_dataset first.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to {output_dir}...")
        self.dataset.save_to_disk(output_dir)
        logger.info("Dataset saved successfully")
    
    def load_dataset(self, input_dir: str = "data/processed") -> DatasetDict:
        """
        Load preprocessed dataset from disk.
        
        Args:
            input_dir: Directory containing saved dataset
            
        Returns:
            Loaded DatasetDict
        """
        logger.info(f"Loading dataset from {input_dir}...")
        self.dataset = DatasetDict.load_from_disk(input_dir)
        logger.info("Dataset loaded successfully")
        return self.dataset


def main():
    """Main function for data loading and exploration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AG News Data Loader")
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--explore', action='store_true', help='Explore dataset')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset')
    parser.add_argument('--save', action='store_true', help='Save processed dataset')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    loader = AGNewsDataLoader(args.config)
    
    if args.download:
        loader.download_dataset()
    
    if args.explore:
        loader.explore_dataset()
    
    if args.preprocess:
        loader.preprocess_dataset(loader.config['model']['name'])
    
    if args.save:
        loader.save_dataset()
    
    if not any([args.download, args.explore, args.preprocess, args.save]):
        # Default: download and explore
        loader.download_dataset()
        loader.explore_dataset()


if __name__ == "__main__":
    main()
