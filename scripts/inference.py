"""
Inference Module for AG News Classification

This module handles:
- Single and batch predictions
- Confidence score calculation
- Probability distribution
- Production-ready inference utilities
"""

import logging
from typing import Dict, List, Union, Optional
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsClassifier:
    """Production-ready news classifier for AG News categories."""
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.class_names = self.config['class_labels']
        self.max_length = self.config['model']['max_length']
        
        self._load_model()
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def _load_model(self) -> None:
        """Load pre-trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _preprocess_text(self, text: str) -> Dict:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokenized inputs
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def predict(self, text: str) -> Dict:
        """
        Predict category for single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with predictions and probabilities
        """
        inputs = self._preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get prediction
        pred_class_id = np.argmax(probabilities)
        pred_class_name = self.class_names[int(pred_class_id)]
        confidence = float(probabilities[pred_class_id])
        
        # Create probability distribution
        prob_dict = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        return {
            'text': text,
            'category': pred_class_name,
            'category_id': int(pred_class_id),
            'confidence': confidence,
            'probabilities': prob_dict,
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict categories for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
            
            # Process each prediction
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                pred_probs = probabilities[j]
                pred_class_id = np.argmax(pred_probs)
                pred_class_name = self.class_names[int(pred_class_id)]
                confidence = float(pred_probs[pred_class_id])
                
                prob_dict = {
                    self.class_names[k]: float(pred_probs[k])
                    for k in range(len(self.class_names))
                }
                
                results.append({
                    'text': text,
                    'category': pred_class_name,
                    'category_id': int(pred_class_id),
                    'confidence': confidence,
                    'probabilities': prob_dict,
                })
        
        return results
    
    def predict_with_threshold(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Optional[Dict]:
        """
        Predict with confidence threshold.
        
        Args:
            text: Input text
            threshold: Confidence threshold
            
        Returns:
            Prediction dictionary if confidence >= threshold, else None
        """
        result = self.predict(text)
        
        if result['confidence'] >= threshold:
            return result
        else:
            logger.warning(
                f"Prediction confidence {result['confidence']:.4f} "
                f"below threshold {threshold}"
            )
            return None
    
    def get_top_k_predictions(self, text: str, k: int = 3) -> List[Dict]:
        """
        Get top-k predictions.
        
        Args:
            text: Input text
            k: Number of top predictions
            
        Returns:
            List of top-k predictions sorted by confidence
        """
        result = self.predict(text)
        
        # Sort probabilities
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_k = sorted_probs[:k]
        
        return [
            {
                'category': category,
                'confidence': float(prob)
            }
            for category, prob in top_k
        ]


def main():
    """Main function for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AG News Inference")
    parser.add_argument('--model-path', default='models/best_model.pt', help='Model path')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--text', help='Input text for prediction')
    parser.add_argument('--batch', action='store_true', help='Batch prediction mode')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--top-k', type=int, default=3, help='Top-k predictions')
    
    args = parser.parse_args()
    
    classifier = NewsClassifier(args.model_path, args.config)
    
    if args.text:
        if args.batch:
            # Batch prediction
            texts = [args.text]  # Can be extended with multiple texts
            results = classifier.predict_batch(texts)
            for result in results:
                print(f"\nText: {result['text']}")
                print(f"Category: {result['category']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("Probabilities:")
                for cat, prob in result['probabilities'].items():
                    print(f"  {cat}: {prob:.4f}")
        else:
            # Single prediction
            result = classifier.predict(args.text)
            print(f"\nText: {result['text']}")
            print(f"Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            for cat, prob in result['probabilities'].items():
                print(f"  {cat}: {prob:.4f}")
            
            # Top-k predictions
            print(f"\nTop-{args.top_k} Predictions:")
            top_k = classifier.get_top_k_predictions(args.text, args.top_k)
            for i, pred in enumerate(top_k, 1):
                print(f"  {i}. {pred['category']}: {pred['confidence']:.4f}")
    else:
        # Interactive mode
        print("AG News Classifier - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            text = input("Enter news text: ").strip()
            if text.lower() == 'quit':
                break
            
            result = classifier.predict(text)
            print(f"\nCategory: {result['category']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            for cat, prob in result['probabilities'].items():
                print(f"  {cat}: {prob:.4f}")
            print()


if __name__ == "__main__":
    main()
