# Kaggle Competition Workflow Setup (using uv)

This guide outlines a structured approach for participating in multiple Kaggle competitions, following best practices from top Kagglers.

## Project Structure

```
kaggle/
├── base-requirements.txt  # Common packages across competitions
├── .venv-base/            # Base virtual environment
└── competitions/
    ├── competition-A/
    │   ├── requirements.txt  # Competition-specific packages
    │   ├── configs/          # Hydra configuration files
    │   ├── src/              # Source code
    │   ├── notebooks/        # Jupyter notebooks
    │   └── ...
    ├── competition-B/
    │   ├── requirements.txt
    │   └── ...
    └── ...
```

## Setup Instructions

### 1. Install UV

UV is a faster alternative to pip for Python package management.

```bash
# Install uv using pip (if you don't have it yet)
pip install uv

# Or on macOS with Homebrew
# brew install uv
```

### 2. Create Base Project Structure

```bash
# Create parent directory
mkdir -p kaggle/competitions

# Navigate to parent directory
cd kaggle
```

### 3. Create Base Virtual Environment

```bash
# Create base virtual environment
uv venv .venv-base

# Activate the base environment
source .venv-base/bin/activate  # On Linux/macOS
# OR
# .venv-base\Scripts\activate  # On Windows
```

### 4. Install Base Dependencies

Create a base requirements file:

```bash
cat > base-requirements.txt << EOL
numpy
pandas
scikit-learn
torch
torchvision
hydra-core
omegaconf
neptune-client
matplotlib
seaborn
tqdm
ipykernel
EOL

# Install base requirements using uv
uv add -r base-requirements.txt
```

### 5. Register Jupyter Kernel (Optional)

If you plan to use Jupyter notebooks:

```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=kaggle-base
```

### 6. Setup Neptune.ai for Experiment Tracking

```bash
# Create a Neptune credentials file
mkdir -p ~/.neptune

# Create the config file (you'll need to add your actual API token)
cat > ~/.neptune/neptune.yaml << EOL
api_token: YOUR_API_TOKEN
project: your-username/kaggle-competitions
EOL

# Make sure the file has appropriate permissions
chmod 600 ~/.neptune/neptune.yaml
```

## Starting a New Competition

When you want to participate in a new competition:

```bash
# Make sure you're in the kaggle directory
cd ~/kaggle  # Adjust path as needed

# Activate the base environment if not already activated
source .venv-base/bin/activate  # On Linux/macOS
# OR
# .venv-base\Scripts\activate  # On Windows

# Create new competition directory
mkdir -p competitions/new-competition-name
cd competitions/new-competition-name

# Create competition structure
mkdir -p configs data/{raw,processed} src/{data,models,training,utils} notebooks scripts
```

### Create Competition-specific Requirements

```bash
# Create competition-specific requirements file
cat > requirements.txt << EOL
# Add competition-specific packages here
# Examples:
# lightgbm
# catboost
# xgboost
EOL

# Install competition-specific requirements
uv add -r requirements.txt
```

### Create Basic Configuration

```bash
# Create a basic Hydra configuration file
mkdir -p configs

cat > configs/config.yaml << EOL
dataset:
  name: "competition_data"
  subset_fraction: 0.2  # Use 20% for quick experiments

model:
  architecture: "efficientnet_b0"  # Start with smaller models
  pretrained: true

training:
  batch_size: 32
  epochs: 5  # Start with fewer epochs for quick iterations
  mixed_precision: true  # Use mixed precision for faster training
  optimizer:
    name: "adam"
    lr: 0.001

experiment:
  name: "exp_001"
  tags: ["baseline", "initial_test"]
EOL
```

## Following Top Kaggler Workflow

### 1. Automate Your Workflow

Create scripts for each step of the pipeline:

```bash
# Create script templates
touch src/data/preprocess.py
touch src/models/model.py
touch src/training/train.py
touch src/utils/metrics.py
touch scripts/run_experiment.py
```

### 2. Prioritize Smart Experiments

Use small data subsets and fast iterations:

```python
# Example in scripts/run_experiment.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config")
def run_experiment(cfg: DictConfig):
    # Load subset of data based on config
    # Train model quickly
    # Log results
    pass

if __name__ == "__main__":
    run_experiment()
```

### 3. Track Everything

Set up Neptune.ai for experiment tracking:

```python
# Example in src/training/train.py
import neptune

def train_with_tracking(model, data, cfg):
    # Initialize Neptune run
    run = neptune.init_run(
        project="username/kaggle-competition-name",
        tags=cfg.experiment.tags,
    )
    
    # Log parameters
    run["parameters"] = cfg.model
    
    # Train model and log metrics
    # ...
    
    return model, run
```

### 4. Use Efficient Training Setups

Implement mixed precision training:

```python
# Example in src/training/train.py
from torch.cuda.amp import autocast, GradScaler

def train_step(model, data, optimizer, scaler):
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss
```

### 5. Submit Less, Validate More

Implement robust cross-validation:

```python
# Example in src/utils/validation.py
from sklearn.model_selection import StratifiedKFold

def cross_validate(model_fn, data, cfg, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data.X, data.y)):
        # Train on this fold
        # Evaluate and store score
        # ...
        
    return scores
```

## Adding New Packages Later

If you need to add more packages during the competition:

```bash
# Add a single package
uv add package-name

# Add multiple packages
uv add package1 package2 package3
```

## Synchronizing Requirements

To ensure your requirements.txt file matches your actual environment:

```bash
# Create/update requirements.txt based on installed packages
uv pip freeze > requirements.txt
```

---

This workflow maximizes efficiency for Kaggle competitions by:
1. Reusing common dependencies across competitions
2. Automating repetitive tasks
3. Enabling rapid experimentation
4. Tracking all experiments systematically
5. Focusing on thorough validation

By following this approach, you'll be able to maximize your experiments per time spent, which is the key principle recommended by top Kagglers.