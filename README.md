# Production-Grade NLP Project: AG News Text Classification

A comprehensive machine learning project demonstrating best practices for fine-tuning transformer models on the AG News dataset. This project includes data pipeline, model training with hyperparameter optimization, evaluation, interactive dashboard, and production-ready API serving.

## Project Overview

**Objective**: Fine-tune DistilBERT on AG News dataset for multi-class news classification (World, Sports, Business, Sci/Tech).

**Dataset**: AG News - 120,000 news articles across 4 categories

**Model**: DistilBERT (distilbert-base-uncased) - lightweight, fast, and production-ready

**Key Features**:
- Complete data pipeline with preprocessing and validation
- Hyperparameter optimization using Optuna
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics and visualizations
- Weights & Biases experiment tracking
- Interactive web dashboard for model testing
- FastAPI server for production deployment
- Docker containerization
- Comprehensive logging and monitoring

## Project Structure

```
nlp_ag_news_project/
├── data/                    # Dataset storage
│   ├── raw/                # Original AG News data
│   └── processed/          # Preprocessed data
├── models/                 # Trained models and checkpoints
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Production Python scripts
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation
│   ├── inference.py       # Inference utilities
│   └── api_server.py      # FastAPI server
├── config/                 # Configuration files
│   └── config.yaml        # Hyperparameters and settings
├── logs/                   # Training logs and outputs
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB RAM minimum (16GB recommended)

### Installation Steps

```bash
# Clone or navigate to project directory
cd /home/ubuntu/nlp_ag_news_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download AG News dataset
python scripts/data_loader.py --download
```

## Quick Start

### 1. Data Exploration
```bash
python scripts/data_loader.py --explore
```

### 2. Train Model
```bash
python scripts/train.py --config config/config.yaml --epochs 5 --batch-size 32
```

### 3. Evaluate Model
```bash
python scripts/evaluate.py --model-path models/best_model.pt
```

### 4. Run Inference
```bash
python scripts/inference.py --text "Breaking news about technology" --model-path models/best_model.pt
```

### 5. Start API Server
```bash
python scripts/api_server.py --port 8000
```

## Model Architecture & Training

**Base Model**: DistilBERT (6 layers, 66M parameters)

**Fine-tuning Strategy**:
- Learning rate: 2e-5 (optimized via Optuna)
- Batch size: 32 (gradient accumulation for larger effective batch)
- Optimizer: AdamW with weight decay
- Scheduler: Linear warmup + cosine annealing
- Early stopping: Patience of 3 epochs
- Max sequence length: 512 tokens

**Training Pipeline**:
1. Load pre-trained DistilBERT tokenizer and model
2. Tokenize and preprocess AG News data
3. Create DataLoader with dynamic batching
4. Fine-tune with mixed precision training (FP16)
5. Validate on development set
6. Save best checkpoint based on validation accuracy
7. Evaluate on test set

## Key Results

**Expected Performance**:
- Accuracy: ~92-94% on test set
- F1-Score: ~0.92-0.94 (macro-averaged)
- Inference time: ~50-100ms per sample (CPU)
- Model size: ~268MB (DistilBERT)

**Training Efficiency**:
- Training time: ~2-3 hours on GPU / ~8-10 hours on CPU
- Memory usage: ~4GB GPU / ~8GB CPU

## Advanced Features

### Hyperparameter Optimization
Uses Optuna to automatically search for optimal hyperparameters:
- Learning rate: [1e-5, 5e-5]
- Batch size: [16, 32, 64]
- Warmup steps: [0, 500, 1000]
- Weight decay: [0.0, 0.01, 0.1]

### Experiment Tracking
Weights & Biases integration for:
- Real-time training metrics
- Hyperparameter logging
- Model checkpoints
- Comparison across runs

### Interactive Dashboard
Web interface for:
- Real-time model testing
- Confidence score visualization
- Batch prediction
- Performance metrics display

### Production API
FastAPI server with:
- RESTful endpoints for inference
- Batch processing support
- Request validation
- Response caching
- Health checks

## Usage Examples

### Python API
```python
from scripts.inference import NewsClassifier

classifier = NewsClassifier(model_path="models/best_model.pt")
result = classifier.predict("Apple announces new iPhone 15")
print(result)
# Output: {'category': 'Sci/Tech', 'confidence': 0.98, 'probabilities': {...}}
```

### REST API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone 15"}'

# Response:
# {
#   "category": "Sci/Tech",
#   "confidence": 0.98,
#   "probabilities": {
#     "World": 0.01,
#     "Sports": 0.00,
#     "Business": 0.01,
#     "Sci/Tech": 0.98
#   }
# }
```

## Best Practices Implemented

### Data Handling
- Stratified train/validation/test splits
- Data augmentation for imbalanced classes
- Tokenization with proper padding and truncation
- Efficient data loading with PyTorch DataLoader

### Model Training
- Mixed precision training for efficiency
- Gradient accumulation for larger effective batch sizes
- Learning rate scheduling and warmup
- Early stopping to prevent overfitting
- Checkpoint saving and recovery

### Evaluation
- Multiple metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Per-class performance analysis
- Cross-validation for robustness

### Code Quality
- Type hints throughout codebase
- Comprehensive logging
- Configuration management
- Error handling and validation
- Unit tests for critical functions

### Deployment
- Docker containerization
- Environment variable management
- Health check endpoints
- Request/response validation
- Rate limiting and caching

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model Configuration
model:
  name: "distilbert-base-uncased"
  num_classes: 4
  max_length: 512

# Training Configuration
training:
  learning_rate: 2e-5
  batch_size: 32
  epochs: 5
  warmup_steps: 500
  weight_decay: 0.01

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42

# Optimization
optimization:
  use_mixed_precision: true
  gradient_accumulation_steps: 1
  early_stopping_patience: 3
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch-size 16`
- Enable gradient accumulation: `--gradient-accumulation-steps 2`
- Use CPU: `--device cpu`

### Slow Training
- Use GPU: Ensure CUDA is available
- Increase batch size (if memory allows)
- Use mixed precision: `--mixed-precision`

### Poor Model Performance
- Increase training epochs
- Adjust learning rate
- Check data preprocessing
- Verify data quality and labels

## References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [AG News Dataset](https://www.kaggle.com/amritpal/ag-news)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please follow:
- PEP 8 code style
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features

## Contact & Support

For questions or issues, please open a GitHub issue or contact the project maintainer.
