"""
Comprehensive utility functions for the ORF prediction project.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import json
import pickle
import shutil
import time
import psutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
from datetime import datetime
import warnings
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# RANDOM SEED AND REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 524):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")


def get_random_state() -> Dict[str, Any]:
    """Get current random state for all libraries."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_random_state(state: Dict[str, Any]):
    """Restore random state for all libraries."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


# =============================================================================
# DEVICE AND HARDWARE MANAGEMENT
# =============================================================================

def get_device(prefer_gpu: bool = True, gpu_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for computation with detailed info.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        gpu_id: Specific GPU ID to use (None for auto-select)
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{gpu_id}")
            else:
                print(f"Warning: GPU {gpu_id} not available, using GPU 0")
                device = torch.device("cuda:0")
        else:
            device = torch.device("cuda")
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1e9
            cached = torch.cuda.memory_reserved(device) / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        device = torch.device("cpu")
        cpu_count = torch.get_num_threads()
        print(f"Using CPU with {cpu_count} threads")
    
    return device


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total': psutil.virtual_memory().total / 1e9,
        'memory_available': psutil.virtual_memory().available / 1e9,
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'python_version': '.'.join(map(str, os.sys.version_info[:3])),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_memory': [torch.cuda.get_device_properties(i).total_memory / 1e9 
                          for i in range(torch.cuda.device_count())]
        })
    
    return info


def memory_cleanup():
    """Clean up GPU memory and run garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# FILE AND PATH UTILITIES
# =============================================================================

def create_directory(path: Union[str, Path], exist_ok: bool = True, parents: bool = True):
    """Create directory with robust error handling."""
    path = Path(path)
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except PermissionError:
        raise PermissionError(f"Permission denied creating directory: {path}")
    except OSError as e:
        raise OSError(f"Error creating directory {path}: {e}")

def backup_file(filepath: Union[str, Path], backup_dir: Optional[str] = None) -> Path:
    """Create backup of a file with timestamp."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if backup_dir is None:
        backup_dir = filepath.parent / "backups"
    
    backup_dir = Path(backup_dir)
    create_directory(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
    
    shutil.copy2(filepath, backup_path)
    return backup_path


def find_files(directory: Union[str, Path], pattern: str = "*", 
               recursive: bool = True) -> List[Path]:
    """Find files matching pattern in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive file information."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    stat = filepath.stat()
    return {
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'accessed': datetime.fromtimestamp(stat.st_atime),
        'suffix': filepath.suffix,
        'stem': filepath.stem,
        'parent': str(filepath.parent),
        'absolute_path': str(filepath.absolute())
    }


def count_lines_in_file(filepath: Union[str, Path]) -> int:
    """Count lines in a file efficiently."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def calculate_model_size(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Calculate comprehensive model size information."""

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_count = sum(b.numel() for b in model.buffers())
    
    # Estimate memory usage (assuming float32)
    param_size_mb = param_count * 4 / (1024 * 1024)
    buffer_size_mb = buffer_count * 4 / (1024 * 1024)
    total_size_mb = param_size_mb + buffer_size_mb
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': param_count - trainable_params,
        'buffers': buffer_count,
        'param_size_mb': param_size_mb,
        'buffer_size_mb': buffer_size_mb,
        'total_size_mb': total_size_mb
    }


def print_model_info(model: nn.Module, model_name: str):
    """Print comprehensive model information."""
    print(f"\n{'='*60}")
    print(f"MODEL INFO: {model_name}")
    print(f"{'='*60}")
    
    size_info = calculate_model_size(model)
    
    print(f"Parameters:")
    print(f"  Total: {size_info['total_parameters']:,}")
    print(f"  Trainable: {size_info['trainable_parameters']:,}")
    print(f"  Non-trainable: {size_info['non_trainable_parameters']:,}")
    print(f"  Buffers: {size_info['buffers']:,}")
    
    print(f"\nMemory Usage (estimated):")
    print(f"  Parameters: {size_info['param_size_mb']:.2f} MB")
    print(f"  Buffers: {size_info['buffer_size_mb']:.2f} MB")
    print(f"  Total: {size_info['total_size_mb']:.2f} MB")
    
    # Model architecture summary
    print(f"\nArchitecture:")
    print(model)


def get_model_summary(model: nn.Module) -> str:
    """Get model summary as string."""
    total_params = count_parameters(model)
    model_size = calculate_model_size(model)
    
    summary = f"""
Model Summary:
- Total Parameters: {total_params:,}
- Trainable Parameters: {model_size['trainable_parameters']:,}
- Model Size: {model_size['total_size_mb']:.2f} MB
    """
    return summary


def freeze_model(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze model parameters."""
    for param in model.parameters():
        param.requires_grad = not freeze
    print(f"Model parameters {'frozen' if freeze else 'unfrozen'}")


def initialize_weights(model: nn.Module, method: str = 'xavier'):
    """Initialize model weights using specified method."""
    def init_func(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif method == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_func)
    print(f"Model weights initialized using {method} method")


# =============================================================================
# DATA UTILITIES
# =============================================================================

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data quality issues in DataFrame."""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': {}
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'memory_usage_mb': df[col].memory_usage(deep=True) / 1024 / 1024
        }
        
        # Add type-specific info
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            })
        elif df[col].dtype == 'object':
            col_info.update({
                'avg_length': df[col].astype(str).str.len().mean(),
                'max_length': df[col].astype(str).str.len().max()
            })
        
        quality_report['columns'][col] = col_info
    
    return quality_report


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True,
                   fill_na_method: str = 'drop') -> pd.DataFrame:
    """Clean DataFrame with various options."""
    df_clean = df.copy()
    
    # Remove duplicates
    if drop_duplicates:
        before_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after_count = len(df_clean)
        if before_count != after_count:
            print(f"Removed {before_count - after_count} duplicate rows")
    
    # Handle missing values
    if fill_na_method == 'drop':
        before_count = len(df_clean)
        df_clean = df_clean.dropna()
        after_count = len(df_clean)
        if before_count != after_count:
            print(f"Removed {before_count - after_count} rows with missing values")
    elif fill_na_method == 'forward':
        df_clean = df_clean.fillna(method='ffill')
    elif fill_na_method == 'backward':
        df_clean = df_clean.fillna(method='bfill')
    elif fill_na_method == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    return df_clean


def split_dataframe(df: pd.DataFrame, ratios: List[float], 
                   shuffle: bool = True, random_state: int = 42) -> List[pd.DataFrame]:
    """Split DataFrame into multiple parts with specified ratios."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    splits = []
    start_idx = 0
    
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:  # Last split gets remaining data
            end_idx = len(df)
        else:
            end_idx = start_idx + int(len(df) * ratio)
        
        splits.append(df.iloc[start_idx:end_idx].copy())
        start_idx = end_idx
    
    return splits


def get_transcript_col(df):
        transcript_col = None
        for col in df.columns:
            if 'transcript' in col.lower() and 'id' in col.lower():
                transcript_col = col
                break
        if transcript_col is None:
            raise ValueError("Could not find transcript ID column in annotations")
        return transcript_col

# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for data."""
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean - h, mean + h


def bootstrap_confidence_interval(data: np.ndarray, statistic_func: callable = np.mean,
                                n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return np.percentile(bootstrap_stats, lower_percentile), np.percentile(bootstrap_stats, upper_percentile)


def effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    return (mean1 - mean2) / pooled_std


# =============================================================================
# TIMING AND PERFORMANCE
# =============================================================================

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"{self.name} completed in {self.elapsed:.2f} seconds")
    
    def get_elapsed(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time if self.start_time else 0
        return self.elapsed


@contextmanager
def memory_monitor(name: str = "Operation"):
    """Monitor memory usage during operation."""
    import tracemalloc
    
    tracemalloc.start()
    start_memory = psutil.virtual_memory().used / 1024 / 1024
    
    print(f"Starting {name} - Initial memory: {start_memory:.1f} MB")
    
    try:
        yield
    finally:
        current_memory = psutil.virtual_memory().used / 1024 / 1024
        memory_diff = current_memory - start_memory
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"{name} completed:")
        print(f"  Memory change: {memory_diff:+.1f} MB")
        print(f"  Peak traced memory: {peak / 1024 / 1024:.1f} MB")


def estimate_training_time(num_epochs: int, samples_per_epoch: int,
                          batch_size: int, time_per_batch: float = 0.1) -> str:
    """Estimate training time based on parameters."""
    batches_per_epoch = np.ceil(samples_per_epoch / batch_size)
    total_batches = batches_per_epoch * num_epochs
    total_seconds = total_batches * time_per_batch
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =============================================================================
# LOGGING AND EXPERIMENT TRACKING
# =============================================================================

def setup_logging(log_file: Optional[str] = None, level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('orf_prediction')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_experiment(config, model_name: str, results: Dict[str, Any],
                  log_file: str = None):
    """Log experiment details to file."""
    if log_file is None:
        log_file = f"experiments_{datetime.now().strftime('%Y%m%d')}.log"
    
    experiment_info = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'config': vars(config) if hasattr(config, '__dict__') else config,
        'results': results,
        'system_info': get_system_info()
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(experiment_info, indent=2, default=str) + '\n\n')


class ExperimentTracker:
    """Track experiments with automatic versioning."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        create_directory(self.experiment_dir)
        
        self.current_experiment = None
        self.experiment_data = {}
    
    def start_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """Start a new experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment = f"{name}_{timestamp}"
        
        experiment_path = self.experiment_dir / self.current_experiment
        create_directory(experiment_path)
        
        self.experiment_data = {
            'name': name,
            'timestamp': timestamp,
            'config': config,
            'path': str(experiment_path),
            'results': {},
            'artifacts': []
        }
        
        # Save initial config
        with open(experiment_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"Started experiment: {self.current_experiment}")
        return self.current_experiment
    
    def log_result(self, key: str, value: Any):
        """Log a result to current experiment."""
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")
        
        self.experiment_data['results'][key] = value
        
        # Save updated results
        experiment_path = Path(self.experiment_data['path'])
        with open(experiment_path / 'results.json', 'w') as f:
            json.dump(self.experiment_data['results'], f, indent=2, default=str)
    
    def save_artifact(self, artifact_path: str, description: str = ""):
        """Save an artifact to current experiment."""
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")
        
        artifact_path = Path(artifact_path)
        experiment_path = Path(self.experiment_data['path'])
        
        if artifact_path.exists():
            # Copy artifact to experiment directory
            dest_path = experiment_path / 'artifacts' / artifact_path.name
            create_directory(dest_path.parent)
            shutil.copy2(artifact_path, dest_path)
            
            self.experiment_data['artifacts'].append({
                'original_path': str(artifact_path),
                'saved_path': str(dest_path),
                'description': description
            })
    
    def finish_experiment(self):
        """Finish current experiment."""
        if self.current_experiment is None:
            return
        
        experiment_path = Path(self.experiment_data['path'])
        
        # Save final experiment data
        with open(experiment_path / 'experiment.json', 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)
        
        print(f"Finished experiment: {self.current_experiment}")
        self.current_experiment = None
        self.experiment_data = {}


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def setup_plotting_style(style: str = 'seaborn', figsize: Tuple[int, int] = (10, 6)):
    """Setup matplotlib plotting style."""
    plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def save_figure(fig, filepath: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save matplotlib figure with good defaults."""
    filepath = Path(filepath)
    create_directory(filepath.parent)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to: {filepath}")


def plot_metrics_comparison(metrics_dict: Dict[str, List[float]], 
                          title: str = "Metrics Comparison",
                          save_path: Optional[str] = None) -> plt.Figure:
    """Plot comparison of multiple metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(metrics_dict))
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.bar(i, mean_val, yerr=std_val, capsize=5, alpha=0.7, label=metric_name)
        ax.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}±{std_val:.3f}', 
                ha='center', va='bottom')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_dict.keys(), rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(config) -> bool:
    """Comprehensive configuration validation."""
    required_attrs = [
        'input_features', 'num_features', 'seed', 'num_epochs',
        'batch_size', 'learning_rate', 'n_splits'
    ]
    
    # Check required attributes
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Config missing required attribute: {attr}")
    
    # Type validation
    if not isinstance(config.input_features, int) or config.input_features <= 0:
        raise ValueError("input_features must be a positive integer")
    
    if not isinstance(config.num_features, int) or config.num_features <= 0:
        raise ValueError("num_features must be a positive integer")
    
    if not isinstance(config.batch_size, int) or config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    
    if not isinstance(config.learning_rate, (int, float)) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number")
    
    # Path validation
    if hasattr(config, 'input_path') and not os.path.exists(config.input_path):
        warnings.warn(f"Input path does not exist: {config.input_path}")
    
    return True


# =============================================================================
# SERIALIZATION UTILITIES
# =============================================================================

def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file."""
    filepath = Path(filepath)
    create_directory(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"Object saved to: {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str, indent: int = 2):
    """Save object to JSON file."""
    filepath = Path(filepath)
    create_directory(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=indent, default=str)
    
    print(f"JSON saved to: {filepath}")


def load_json(filepath: str) -> Any:
    """Load object from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Advanced progress tracking with ETA and speed estimates."""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = self.start_time
    
    def update(self, step: Optional[int] = None, message: str = ""):
        """Update progress with optional message."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            eta = avg_time_per_step * (self.total_steps - self.current_step)
            speed = 1 / avg_time_per_step if avg_time_per_step > 0 else 0
        else:
            eta = 0
            speed = 0
        
        progress = self.current_step / self.total_steps * 100
        
        eta_str = f"{int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}"
        
        status = f"\r{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps}) "
        status += f"| ETA: {eta_str} | Speed: {speed:.2f} it/s"
        
        if message:
            status += f" | {message}"
        
        print(status, end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete
            total_time = elapsed
            print(f"Completed in {int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_step < self.total_steps:
            print()  # Ensure new line if not completed


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def format_number(num: Union[int, float], decimals: int = 4) -> str:
    """Format a number with specified decimal places."""
    if isinstance(num, int):
        return f"{num:,}"
    else:
        return f"{num:,.{decimals}f}"


def format_bytes(bytes_size: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m {int(seconds%60)}s"


# Make commonly used functions available at module level
__all__ = [
    'set_seed', 'get_device', 'memory_cleanup', 'create_directory',
    'count_parameters', 'print_model_info', 'Timer', 'ProgressTracker',
    'setup_logging', 'validate_config', 'save_pickle', 'load_pickle',
    'save_json', 'load_json', 'format_number', 'format_bytes', 'format_time'
]


import numpy as np
import torch
import math
from typing import Optional, Union, Tuple

def create_sinusoidal_positional_encoding(seq_length: int, d_model: int, 
                                        max_len: int = 100000) -> np.ndarray:
    """
    Create sinusoidal positional encoding like in Transformer models.
    
    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the positional encoding
        max_len: Maximum sequence length for frequency calculation
        
    Returns:
        Positional encoding array of shape (seq_length, d_model)
    """
    pe = np.zeros((seq_length, d_model))
    position = np.arange(0, seq_length).reshape(-1, 1)
    
    # Create frequency terms
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(max_len) / d_model))
    
    # Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices  
    if d_model > 1:
        pe[:, 1::2] = np.cos(position * div_term[:d_model//2 + d_model%2])
    
    return pe.astype(np.float32)

def create_linear_positional_encoding(seq_length: int, d_model: int = 1, 
                                    normalize: bool = True) -> np.ndarray:
    """
    Create simple linear positional encoding.
    
    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the positional encoding (usually 1)
        normalize: Whether to normalize positions to [0, 1]
        
    Returns:
        Positional encoding array of shape (seq_length, d_model)
    """
    positions = np.arange(seq_length, dtype=np.float32)
    
    if normalize:
        positions = positions / (seq_length - 1) if seq_length > 1 else positions
    
    if d_model == 1:
        return positions.reshape(-1, 1)
    else:
        # Repeat the same position for all dimensions
        return np.tile(positions.reshape(-1, 1), (1, d_model))

def create_relative_positional_encoding(seq_length: int, d_model: int = 2,
                                      window_size: int = 100) -> np.ndarray:
    """
    Create relative positional encoding based on distance from sequence center.
    
    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the positional encoding
        window_size: Window size for relative positioning
        
    Returns:
        Positional encoding array of shape (seq_length, d_model)
    """
    center = seq_length // 2
    positions = np.arange(seq_length, dtype=np.float32)
    
    # Distance from center
    distances = np.abs(positions - center)
    
    # Normalize distances
    max_distance = max(center, seq_length - center)
    normalized_distances = distances / max_distance if max_distance > 0 else distances
    
    pe = np.zeros((seq_length, d_model))
    
    if d_model >= 1:
        pe[:, 0] = normalized_distances  # Distance from center
    
    if d_model >= 2:
        pe[:, 1] = positions / (seq_length - 1) if seq_length > 1 else 0  # Absolute position
    
    if d_model >= 3:
        # Add sinusoidal components for higher dimensions
        pe[:, 2:] = create_sinusoidal_positional_encoding(seq_length, d_model - 2)
    
    return pe.astype(np.float32)

def create_codon_positional_encoding(seq_length: int, d_model: int = 3) -> np.ndarray:
    """
    Create codon-aware positional encoding (position within codon + codon index).
    
    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the positional encoding
        
    Returns:
        Positional encoding array of shape (seq_length, d_model)
    """
    pe = np.zeros((seq_length, d_model))
    
    for i in range(seq_length):
        codon_index = i // 3  # Which codon
        position_in_codon = i % 3  # Position within codon (0, 1, 2)
        
        if d_model >= 1:
            pe[i, 0] = position_in_codon / 2.0  # Normalize to [0, 1]
        
        if d_model >= 2:
            pe[i, 1] = codon_index / (seq_length // 3) if seq_length > 3 else 0
        
        if d_model >= 3:
            # One-hot encoding of position within codon
            if position_in_codon < d_model - 2:
                pe[i, 2 + position_in_codon] = 1.0
    
    return pe.astype(np.float32)

class PositionalEncodingConfig:
    """Configuration for positional encoding."""
    
    def __init__(self, encoding_type: str = 'sinusoidal', d_model: int = 2, 
                 normalize: bool = False, **kwargs):
        """
        Initialize positional encoding configuration.
        
        Args:
            encoding_type: Type of encoding ('none', 'linear', 'sinusoidal', 'relative', 'codon')
            d_model: Dimension of positional encoding
            normalize: Whether to normalize positions
            **kwargs: Additional parameters for specific encoding types
        """
        self.encoding_type = encoding_type
        self.d_model = d_model
        self.normalize = normalize
        self.kwargs = kwargs
    
    def create_encoding(self, seq_length: int) -> Optional[np.ndarray]:
        """Create positional encoding for given sequence length."""
        if self.encoding_type == 'none':
            return None
        elif self.encoding_type == 'linear':
            return create_linear_positional_encoding(seq_length, self.d_model, self.normalize)
        elif self.encoding_type == 'sinusoidal':
            return create_sinusoidal_positional_encoding(seq_length, self.d_model, **self.kwargs)
        elif self.encoding_type == 'relative':
            return create_relative_positional_encoding(seq_length, self.d_model, **self.kwargs)
        elif self.encoding_type == 'codon':
            return create_codon_positional_encoding(seq_length, self.d_model)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")