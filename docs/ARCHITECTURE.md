# Architecture

An end-to-end text classification pipeline that fine-tunes DistilBERT on AG News (World, Sports, Business, Sci/Tech) and serves it through a FastAPI endpoint.

## Pipeline stages

```
data_loader.py  ->  train.py  ->  evaluate.py  ->  inference.py / api_server.py
 (download,        (fine-tune,     (metrics,          (single-text prediction,
  preprocess)       Optuna, W&B)    confusion matrix)   HTTP serving)
```

| Script | Role |
| ------ | ---- |
| `scripts/data_loader.py` | Downloads AG News, cleans and tokenizes text, writes `data/raw` and `data/processed`. `--explore` prints dataset statistics. |
| `scripts/train.py` | Fine-tunes `distilbert-base-uncased` with AdamW, linear warmup + cosine schedule, mixed precision, early stopping, and optional Optuna hyperparameter search. Tracks runs in Weights & Biases. Saves the best checkpoint to `models/`. |
| `scripts/evaluate.py` | Loads a checkpoint and reports accuracy, macro F1, per-class metrics, and plots. |
| `scripts/inference.py` | Command-line prediction for a single text. |
| `scripts/api_server.py` | FastAPI app exposing prediction over HTTP. |
| `scripts/quick_explore.py` | Lightweight dataset exploration. |
| `config/config.yaml` | Hyperparameters, paths, and training settings shared by all scripts. |

## Model

- Base: DistilBERT, 6 layers, 66M parameters, max sequence length 512.
- Head: linear classifier over 4 labels.
- Training defaults: lr 2e-5, batch 32 (with gradient accumulation), patience 3 epochs.

## Artifacts and directories

```
data/raw, data/processed   # datasets (gitignored)
models/                    # checkpoints such as best_model.pt
logs/                      # training logs
notebooks/                 # exploration and training_guide.md
```

## Serving

`api_server.py` loads the checkpoint once at startup and exposes a JSON endpoint for classification. The `Dockerfile` and `docker-compose.yml` package the server for deployment.

See `IMPLEMENTATION_GUIDE.md`, `PROJECT_SUMMARY.md`, and `TODO.md` at the repo root for design notes and open items.
