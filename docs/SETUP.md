# Setup

## Prerequisites

- Python 3.8+
- 8 GB RAM minimum (16 GB recommended); CUDA 11+ optional for GPU training

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/data_loader.py --download
```

## Train, evaluate, predict

```bash
python scripts/train.py --config config/config.yaml --epochs 5 --batch-size 32
python scripts/evaluate.py --model-path models/best_model.pt
python scripts/inference.py --text "Breaking news about technology" --model-path models/best_model.pt
```

## Serve

```bash
python scripts/api_server.py --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

## Configuration

Edit `config/config.yaml` for learning rate, batch size, epochs, warmup, max length, and output paths. Command-line flags override the file.

## Experiment tracking

Set `WANDB_API_KEY` in your environment to log runs to Weights & Biases; training runs offline otherwise.
