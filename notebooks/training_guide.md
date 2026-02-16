# AG News Classification - Training & Evaluation Guide

## Overview

This guide walks through the complete training and evaluation pipeline for fine-tuning DistilBERT on the AG News dataset.

## Step 1: Data Preparation

### Download and Explore Dataset

```bash
cd /home/ubuntu/nlp_ag_news_project
python scripts/quick_explore.py
```

**Expected Output:**
- Total samples: 127,600
- Train: 120,000 (94.0%)
- Test: 7,600 (6.0%)
- Perfectly balanced: 25% each class

### Preprocess Dataset

```bash
python scripts/data_loader.py --preprocess --save
```

This will:
- Tokenize all texts using DistilBERT tokenizer
- Create train/validation/test splits (80/10/10)
- Save processed dataset to `data/processed/`

## Step 2: Model Training

### Basic Training

```bash
python scripts/train.py \
  --config config/config.yaml \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --warmup-steps 500
```

### Training with Custom Parameters

```bash
python scripts/train.py \
  --config config/config.yaml \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-5 \
  --warmup-steps 1000
```

### Expected Training Time

- **GPU (NVIDIA A100)**: ~1-2 hours for 5 epochs
- **GPU (NVIDIA V100)**: ~2-3 hours for 5 epochs
- **CPU**: ~8-10 hours for 5 epochs

### Training Features

The training script includes:

**Optimization Techniques:**
- Mixed precision training (FP16) for faster computation
- Gradient accumulation for larger effective batch sizes
- Learning rate warmup and scheduling
- Gradient clipping to prevent exploding gradients
- Weight decay (L2 regularization)

**Monitoring:**
- Real-time training loss logging
- Validation metrics every epoch
- Weights & Biases experiment tracking
- Checkpoint saving for best model

**Early Stopping:**
- Monitors validation accuracy
- Saves best checkpoint automatically
- Stops training if no improvement for 3 epochs

## Step 3: Model Evaluation

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
  --model-path models/best_model.pt \
  --dataset-path data/processed \
  --split test \
  --visualize
```

### Expected Performance

Based on DistilBERT fine-tuning on AG News:

- **Accuracy**: 92-94%
- **Precision**: 0.92-0.94 (macro-averaged)
- **Recall**: 0.92-0.94 (macro-averaged)
- **F1-Score**: 0.92-0.94 (macro-averaged)

### Evaluation Outputs

The script generates:

1. **Confusion Matrix** (`logs/confusion_matrix.png`)
   - Shows per-class prediction accuracy
   - Identifies which classes are confused

2. **Per-Class Metrics** (`logs/per_class_metrics.png`)
   - Precision, Recall, F1-Score for each class
   - Identifies class-specific performance

3. **Confidence Distribution** (`logs/confidence_distribution.png`)
   - Distribution of prediction confidence
   - Separates correct vs. incorrect predictions

## Step 4: Inference

### Single Prediction

```bash
python scripts/inference.py \
  --model-path models/best_model.pt \
  --text "Apple announces new iPhone 15 Pro"
```

**Output:**
```
Category: Sci/Tech
Confidence: 0.98
Probabilities:
  World: 0.01
  Sports: 0.00
  Business: 0.01
  Sci/Tech: 0.98
```

### Batch Prediction

```bash
python scripts/inference.py \
  --model-path models/best_model.pt \
  --batch \
  --text "Multiple news texts..."
```

### Interactive Mode

```bash
python scripts/inference.py --model-path models/best_model.pt
```

Then enter news texts and get predictions interactively.

## Step 5: API Deployment

### Start API Server

```bash
python scripts/api_server.py --port 8000
```

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone 15"}'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Apple announces new iPhone 15", "Manchester United wins 3-0"],
    "batch_size": 32
  }'
```

**Get Categories:**
```bash
curl http://localhost:8000/categories
```

**Model Info:**
```bash
curl http://localhost:8000/model-info
```

## Advanced Features

### Hyperparameter Optimization (Optuna)

Enable in `config/config.yaml`:

```yaml
optuna:
  enabled: true
  n_trials: 20
  sampler: "tpe"
  timeout: 3600
```

Then run training - it will automatically search for optimal hyperparameters.

### Weights & Biases Tracking

Enable in `config/config.yaml`:

```yaml
wandb:
  enabled: true
  project: "ag-news-classification"
```

Then view results at: https://wandb.ai/your-username/ag-news-classification

### Custom Configuration

Edit `config/config.yaml` to customize:

```yaml
# Learning rate scheduling
scheduler:
  type: "cosine"  # or "linear"
  num_cycles: 0.5

# Early stopping
early_stopping:
  enabled: true
  patience: 3
  min_delta: 0.0001

# Data augmentation
augmentation:
  enabled: false
  techniques:
    - "synonym_replacement"
    - "random_insertion"
```

## Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```bash
python scripts/train.py --batch-size 16
```

**Solution 2: Enable gradient accumulation**
Edit `config/config.yaml`:
```yaml
training:
  gradient_accumulation_steps: 2
```

**Solution 3: Use CPU**
```bash
# Edit config/config.yaml
device:
  type: "cpu"
```

### Slow Training

**Solution 1: Use GPU**
Ensure CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution 2: Increase batch size**
```bash
python scripts/train.py --batch-size 64
```

### Poor Model Performance

1. **Check data quality**: Ensure dataset is properly preprocessed
2. **Increase training epochs**: Try 10-15 epochs instead of 5
3. **Adjust learning rate**: Try 1e-5 or 5e-5
4. **Verify labels**: Ensure labels are correctly mapped

## Performance Benchmarks

### Model Size & Speed

| Metric | Value |
|--------|-------|
| Model Parameters | 66M |
| Model Size | 268MB |
| Inference Time (CPU) | 50-100ms |
| Inference Time (GPU) | 10-20ms |
| Max Sequence Length | 512 tokens |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 127,600 |
| Train Samples | 120,000 |
| Test Samples | 7,600 |
| Classes | 4 |
| Class Balance | Perfect (25% each) |
| Average Text Length | ~40 tokens |

## Best Practices

1. **Always validate on a separate set**: Use the validation split during training
2. **Monitor training curves**: Watch for overfitting
3. **Save checkpoints**: Keep the best model checkpoint
4. **Test on diverse data**: Ensure model generalizes well
5. **Use appropriate batch sizes**: Balance speed and memory
6. **Enable mixed precision**: Faster training with minimal accuracy loss
7. **Use learning rate scheduling**: Helps convergence
8. **Monitor with W&B**: Track experiments and compare runs

## Next Steps

1. Fine-tune on custom datasets
2. Implement data augmentation
3. Experiment with different architectures (BERT, RoBERTa)
4. Deploy to production (Docker, Kubernetes)
5. Create monitoring dashboards
6. Implement A/B testing

## References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [AG News Dataset](https://www.kaggle.com/amritpal/ag-news)
- [PyTorch Documentation](https://pytorch.org/docs/)
