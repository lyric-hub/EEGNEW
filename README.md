# EEG Neural Network for Motor Imagery Classification

A deep learning framework for EEG-based Motor Imagery (MI) classification using PyTorch, Hydra, DVC, and MLflow.

## 🚀 Quick Start

# 3. Run the full pipeline
dvc repro

# Or run steps individually:
python scripts/download_data.py   # Download MOABB dataset
python scripts/preprocess.py      # Preprocess EEG data
python scripts/train.py           # Train model
```

## 📁 Project Structure

```
eeg-nn/
├── config/                     # Hydra configuration
│   ├── config.yaml             # Main config
│   ├── data/default.yaml       # Dataset settings
│   ├── model/custom.yaml       # Model hyperparameters
│   └── training/default.yaml   # Training settings
├── src/                        # Source code
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Neural network architectures
│   ├── training/               # Training loop & metrics
│   └── utils/                  # Helper functions
├── scripts/                    # Entry point scripts
│   ├── download_data.py        # Download datasets
│   ├── preprocess.py           # Preprocess EEG data
│   └── train.py                # Train model
├── data/                       # Data directory (DVC tracked)
├── models/                     # Saved checkpoints
├── outputs/                    # Hydra outputs
├── dvc.yaml                    # DVC pipeline
└── pyproject.toml              # Dependencies
```

## ⚙️ Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Override any parameter from CLI:

```bash
# Change learning rate
python scripts/train.py training.optimizer.lr=0.0005

# Change batch size
python scripts/train.py data.batch_size=64

# Run hyperparameter sweep
python scripts/train.py --multirun training.optimizer.lr=0.001,0.0005,0.0001
```

View all options:
```bash
python scripts/train.py --help
```

## 🧠 Developing Your Model

1. **Edit model architecture**: `src/models/custom.py`
2. **Edit model config**: `config/model/custom.yaml`
3. **Test your model**:
   ```bash
   python -c "
   import torch
   from src.models.custom import CustomModel
   model = CustomModel(n_channels=22, n_samples=256, n_classes=4)
   x = torch.randn(8, 22, 256)
   print(f'Output shape: {model(x).shape}')
   print(f'Parameters: {model.count_parameters():,}')
   "
   ```
4. **Train**: `python scripts/train.py`

## 📊 Experiment Tracking

Experiments are tracked with [MLflow](https://mlflow.org/):

```bash
# Start MLflow UI
mlflow ui

# Open http://localhost:5000 in browser
```

## 🔄 DVC Pipeline

```
 +----------+  
 | download |  
 +----------+  
       *       
+------------+ 
| preprocess | 
+------------+ 
       *       
  +-------+    
  | train |    
  +-------+    
```

Commands:
```bash
dvc repro          # Run full pipeline
dvc dag            # View pipeline graph
dvc metrics show   # Show metrics
```

## 🎯 Hyperparameter Tuning

Automated hyperparameter tuning with [Optuna](https://optuna.org/):

```bash
# Run 50 trials
python scripts/tune.py --n-trials 50

# Run with timeout (1 hour)
python scripts/tune.py --n-trials 100 --timeout 3600
```

**Search space** (defined in `scripts/tune.py`):
- Learning rate: `1e-5` to `1e-2` (log scale)
- Weight decay: `1e-6` to `1e-2` (log scale)
- Batch size: `[16, 32, 64]`
- Hidden dim: `[64, 128, 256]`
- Dropout: `0.3` to `0.7`

**Output:**
- `outputs/tuning/best_params.yaml` - Best hyperparameters
- `outputs/tuning/study.db` - SQLite database for analysis

**Apply best params:**
```bash
python scripts/train.py model.hidden_dim=256 training.optimizer.lr=0.0003
```

## 📦 Dependencies

Managed with [Poetry](https://python-poetry.org/):

```bash
poetry add <package>      # Add dependency
poetry install            # Install all dependencies
poetry show               # List installed packages
```

## 📄 License

MIT License
