"""
Training utilities and loss functions for the ORF prediction project.
"""
import os
import json
import hashlib
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import KFold

from dataset import create_mask_from_lengths, ORFDataset, NTv3Dataset, create_data_loaders
from typing import Dict, List
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.stats import pearsonr
from models import get_model
from utils import count_parameters, PositionalEncodingConfig
from pathlib import Path

from loss import MSELoss, MAELoss, HuberLoss, \
                 MulticlassFocalLoss, CrossEntropyLoss,\
                 FocalLoss, BCELoss, \
                 MultiLoss,\
                 compute_zero_output_loss


def _distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_epoch_timing(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _save_fold_manifest(result_path, fold, seed, file_list, train_idx, val_idx):
    manifest_dir = Path(result_path) / "fold_splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"fold_{fold}.tsv"
    rows = []
    for split, indices in (("train", train_idx), ("validation", val_idx)):
        for index in indices:
            filename = file_list[index]
            rows.append((split, filename, filename.removesuffix(".txt").split("_")[0]))
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("split\tfilename\ttranscript\n")
        for split, filename, transcript in rows:
            handle.write(f"{split}\t{filename}\t{transcript}\n")

    validation_transcripts = sorted(
        {transcript for split, _, transcript in rows if split == "validation"}
    )
    digest = hashlib.sha256(
        "".join(f"{name}\n" for name in validation_transcripts).encode()
    ).hexdigest()
    metadata = {
        "fold": fold,
        "seed": seed,
        "file_order": "sorted",
        "train_files": len(train_idx),
        "validation_files": len(val_idx),
        "validation_transcripts": len(validation_transcripts),
        "validation_transcripts_sha256": digest,
    }
    metadata_path = manifest_dir / f"fold_{fold}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _int_env(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _save_checkpoint(path, model, optimizer, loss_history, fold, epoch, global_step, config, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss_history": loss_history,
        "fold": fold,
        "epoch": epoch,
        "global_step": global_step,
        "config": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
        },
    }, path)
    print(f"Saved checkpoint: {path}", flush=True)


def _load_checkpoint(path, model, optimizer, device, scheduler=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    loss_history = checkpoint.get("loss_history", [])
    print(f"Loaded checkpoint: {path} epoch={start_epoch} global_step={global_step}", flush=True)
    return start_epoch, global_step, loss_history


def _build_lr_scheduler(optimizer, num_epochs: int, train_steps_per_epoch: int):
    scheduler_name = os.environ.get("LR_SCHEDULER", "none").strip().lower()
    if scheduler_name in ("", "none", "off", "false", "0"):
        return None, {
            "name": "none",
            "total_updates": 0,
            "warmup_steps": 0,
            "min_lr": 0.0,
            "warmup_lr_init": 0.0,
        }
    if scheduler_name != "cosine":
        raise ValueError("LR_SCHEDULER must be 'none' or 'cosine'")

    total_updates = _int_env("LR_TOTAL_STEPS", max(1, num_epochs * max(1, train_steps_per_epoch)))
    warmup_steps = _int_env("WARMUP_STEPS", 100)
    min_lr = _float_env("MIN_LR", 1e-6)
    warmup_lr_init = _float_env("WARMUP_LR_INIT", min_lr)

    try:
        from timm.scheduler.cosine_lr import CosineLRScheduler
    except ImportError as exc:
        raise ImportError(
            "LR_SCHEDULER=cosine requires timm. Install timm in the container or disable scheduling with LR_SCHEDULER=none."
        ) from exc

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_updates,
        lr_min=min_lr,
        warmup_t=warmup_steps,
        warmup_lr_init=warmup_lr_init,
        t_in_epochs=False,
    )
    return scheduler, {
        "name": "cosine",
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "warmup_lr_init": warmup_lr_init,
    }


def _build_criterion(config) -> MultiLoss:
    lc = config.loss
    bin_target_threshold_env = os.environ.get("BIN_TARGET_THRESHOLD", "").strip()
    if bin_target_threshold_env:
        bin_target_threshold = float(bin_target_threshold_env)

        def binarization_target(batch, outputs):
            targets = batch.get("targets", batch.get("bigwig_targets"))
            return (targets > bin_target_threshold).to(outputs["binarization"].dtype)
    else:
        def binarization_target(batch, outputs):
            return 1.0 - batch["codon_seq"][:, :, 0:1]

    # regression
    if lc.regression.kind == "huber":
        reg_loss = HuberLoss(delta=lc.regression.huber_delta)
    elif lc.regression.kind == "mae":
        reg_loss = MAELoss()
    elif lc.regression.kind == "mse":
        reg_loss = MSELoss()
    else:
        raise ValueError(f"Unsupported regression loss: '{lc.regression.kind}'. Choose from: 'mse', 'mae', 'huber'")

    # binarization
    if lc.binarization.kind == "bce":
        bin_loss = BCELoss()
    elif lc.binarization.kind == "focal":
        bin_loss = FocalLoss(alpha=lc.binarization.focal_alpha, gamma=lc.binarization.focal_gamma)
    else:
        raise ValueError(f"Unsupported binarization loss: '{lc.binarization.kind}'. Choose from: 'bce', 'focal'")

    # classification
    if lc.classification.kind == "cross_entropy":
        cls_loss = CrossEntropyLoss()
    elif lc.classification.kind == "multiclass_focal":
        cls_loss = MulticlassFocalLoss(gamma=lc.classification.gamma)
    else:
        raise ValueError(f"Unsupported classification loss: '{lc.classification.kind}'. Choose from: 'cross_entropy', 'multiclass_focal'")


    return MultiLoss({
            "regression": {
                "loss": reg_loss,
                "pred_key": "regression",
                "target_getter": lambda batch, outputs: batch.get("targets", batch.get("bigwig_targets")),
                "weight": lc.regression.weight,
            },
            "classification": {
                "loss": cls_loss,
                "pred_key": "classification",
                "target_getter": lambda batch, outputs: batch.get("codon_seq", torch.zeros_like(outputs["classification"])),
                "weight": lc.classification.weight,
            },
            "binarization": {
                "loss": bin_loss,
                "pred_key": "binarization",
                "target_getter": binarization_target,
                "weight": lc.binarization.weight,
            },
        })


def train_epoch(config, model_name, model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device,
                global_step: Optional[int] = None,
                step_callback = None,
                max_steps: int = 0) -> float:

    model.train()
    total_loss = 0.0
    num_batches = 0
    total_loss_dict = {}
    scaler = torch.amp.GradScaler("cuda")
    log_every_steps = _int_env("LOG_EVERY_STEPS", 0)
    train_started = time.perf_counter()
    previous_step_finished = train_started

    for batch in train_loader:
        if max_steps > 0 and num_batches >= max_steps:
            break
        optimizer.zero_grad()

        if "NTv3" in config.model:
            batch_data, batch_lengths = batch
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }
            batch_lengths = batch_lengths.to(device)
            mask = create_mask_from_lengths(batch_lengths, batch_on_device["tokens"].shape[1], device)  # ← remove do_unsqueeze=True
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(tokens=batch_on_device["tokens"], conditions=batch_on_device["extra_features"])
                loss, loss_dict = criterion(outputs=outputs, batch=batch_on_device, mask=mask)
        
        else:
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # train_epochの最初のバッチで確認
            batch_lengths = batch_on_device["lengths"]
            mask = create_mask_from_lengths(batch_lengths, batch_on_device["tokens"].shape[1], device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch_on_device["tokens"], mask)
                loss, loss_dict = criterion(outputs=outputs, batch=batch_on_device, mask=mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if global_step is not None:
            global_step += 1
            if step_callback is not None:
                step_callback(global_step)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v.item()  # v is already detached
        num_batches += 1
        step_finished = time.perf_counter()
        step_seconds = step_finished - previous_step_finished
        if _is_rank0() and log_every_steps > 0 and num_batches % log_every_steps == 0:
            elapsed = step_finished - train_started
            print(
                "[train_step] "
                f"time={time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"global_step={global_step if global_step is not None else num_batches} "
                f"epoch_step={num_batches} "
                f"step_time={step_seconds:.3f}s "
                f"elapsed={elapsed:.3f}s "
                f"avg={elapsed / num_batches:.3f}s/step "
                f"gpu_batch_size={config.batch_size}",
                flush=True,
            )
        previous_step_finished = step_finished
    avg_loss_dict = {k: v / num_batches for k, v in total_loss_dict.items()}
    if global_step is not None:
        return total_loss / num_batches, avg_loss_dict, global_step
    return total_loss / num_batches, avg_loss_dict 


def validate_epoch(config, model_name, model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device, max_steps: int = 0) -> float:

    model.eval()
    total_loss = 0.0
    total_loss_dict = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if max_steps > 0 and num_batches >= max_steps:
                break
            if "NTv3" in config.model:
                batch_data, batch_lengths = batch
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_data.items()
                }
                batch_lengths = batch_lengths.to(device)
                mask = create_mask_from_lengths(batch_lengths, batch_on_device["tokens"].shape[1], device)
                outputs = model(tokens=batch_on_device["tokens"], conditions=batch_on_device["extra_features"])
            else:
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch_lengths = batch_on_device["lengths"]
                mask = create_mask_from_lengths(batch_lengths, batch_on_device["tokens"].shape[1], device)
                outputs = model(batch_on_device["tokens"], mask)

            loss, loss_dict = criterion(outputs=outputs, batch=batch_on_device, mask=mask)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v.item()  # v is already detached
            num_batches += 1

    avg_loss_dict = {k: v / num_batches for k, v in total_loss_dict.items()}
    return total_loss / num_batches, avg_loss_dict


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
               num_epochs: int = 100, verbose: bool = True) -> List[Tuple[float, float]]:
    """
    Train the model for multiple epochs.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of epochs to train
        verbose: Whether to print progress
        
    Returns:
        List of (train_loss, val_loss) tuples for each epoch
    """
    loss_history = []

    for epoch in tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        loss_history.append((train_loss, val_loss))
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return loss_history

def plot_loss_history(loss_history, fold_num, zero_output_loss=None, save_path=None):
    df = pd.DataFrame(loss_history)
    epochs = df["epoch"].tolist()

    # Save CSV
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, f"fold_{fold_num}_loss.csv")
        df.to_csv(csv_path, index=False)
        print(f"Loss CSV saved to: {csv_path}")

    weighted_keys = [
        k.replace("train_", "")
        for k in df.columns
        if k.startswith("train_") and k != "train_total" and k != "train_epoch"
    ]
    labels = [k.replace("_weighted", "").replace("_", " ").title() for k in weighted_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split in zip(axes, ["train", "val"]):
        # Stack the weighted components
        stack_data = np.array([df[f"{split}_{k}"].tolist() for k in weighted_keys])  # [n_losses, n_epochs]
        
        ax.stackplot(epochs, stack_data, labels=labels, alpha=0.7)
        
        # Overlay total loss line
        ax.plot(epochs, df[f"{split}_total"], color="black", linewidth=1.5,
                linestyle="--", label="Total")

        if zero_output_loss is not None and split == "val":
            ax.axhline(y=zero_output_loss, color="red", linestyle=":",
                       label=f"Zero baseline ({zero_output_loss:.4f})")

        ax.set_title(f"Fold {fold_num}: {split.title()} Loss Breakdown")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plot_path = os.path.join(save_path, f"fold_{fold_num}_loss.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

def evaluate_model(config, model, data_loader, device):
    model.eval()
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch in data_loader:
            if "NTv3" in config.model:
                batch_input, batch_lengths = batch
                batch_inputs  = batch_input["tokens"].to(device)
                conditions    = batch_input["extra_features"].to(device) 
                batch_targets = batch_input["targets"].to(device)
                batch_lengths = batch_lengths.to(device)
                outputs = model(tokens=batch_inputs, conditions=conditions)["regression"]
            else:
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch_inputs  = batch_on_device["tokens"]
                batch_targets = batch_on_device["targets"]
                batch_lengths = batch_on_device["lengths"]
                mask = create_mask_from_lengths(batch_lengths, batch_inputs.shape[1], device)
                outputs = model(batch_inputs, mask)["regression"]  # extract regression head for eval
            
            for j in range(batch_lengths.shape[0]):
                actual_length = batch_lengths[j].item()
                pred  = outputs[j, :actual_length].cpu().numpy().squeeze()       # [actual_length]
                truth = batch_targets[j, :actual_length].cpu().numpy().squeeze() # [actual_length]

                valid_mask = (truth >= 0) & ~np.isnan(truth)
                if valid_mask.sum() > 0:
                    all_preds.extend(pred[valid_mask])
                    all_truths.extend(truth[valid_mask])
    
    if len(all_preds) == 0:
        return {'pearson_corr': 0.0, 'auc': 0.5, 'pr_auc': 0.0, 'n': 0, 'pos_rate': 0.0}

    all_preds  = np.array(all_preds)
    all_truths = np.array(all_truths)

    corr_coef, _ = pearsonr(all_preds, all_truths) if len(all_preds) > 1 else (0.0, 1.0)

    threshold = getattr(config, 'init_threshold', 0.2)
    binary_labels = (all_truths > threshold).astype(int)
    if len(np.unique(binary_labels)) > 1:
        fpr, tpr, _ = roc_curve(binary_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        pr_auc  = float(average_precision_score(binary_labels, all_preds))
    else:
        roc_auc = 0.5
        pr_auc  = 0.0

    return {
        'pearson_corr': corr_coef,
        'auc':          roc_auc,
        'pr_auc':       pr_auc,
        'n':            int(len(all_preds)),
        'pos_rate':     float(binary_labels.mean()),
    }

def group_files_by_transcript(file_list: List[str]) -> Dict[str, List[int]]:
    """
    Group file indices by transcript base name.
    
    Args:
        file_list: List of filenames
        
    Returns:
        Dictionary mapping transcript base names to lists of file indices
    """
    transcript_groups = {}
    
    for idx, filename in enumerate(file_list):
        # Remove .txt extension
        base_name = filename[:-4] if filename.endswith('.txt') else filename
        transcript_base = base_name.split('_')[0]
        
        if transcript_base not in transcript_groups:
            transcript_groups[transcript_base] = []
        transcript_groups[transcript_base].append(idx)
    
    return transcript_groups

def validate_transcript_groups(transcript_groups: Dict[str, List[int]], expected_replicates: int = 4):
    """
    Validate that each transcript has the expected number of replicates.
    
    Args:
        transcript_groups: Dictionary of transcript groups
        expected_replicates: Expected number of replicates per transcript
    """
    print(f"Found {len(transcript_groups)} unique transcripts")
    
    for transcript_base, indices in transcript_groups.items():
        if len(indices) != expected_replicates:
            print(f"Warning: Transcript '{transcript_base}' has {len(indices)} files, expected {expected_replicates}")
    
    # Print summary
    replicate_counts = {}
    for indices in transcript_groups.values():
        count = len(indices)
        replicate_counts[count] = replicate_counts.get(count, 0) + 1
    
    print("Replicate count distribution:")
    for count, num_transcripts in sorted(replicate_counts.items()):
        print(f"  {count} replicates: {num_transcripts} transcripts")

def create_transcript_stratified_splits(transcript_groups: Dict[str, List[int]], n_splits: int = 10, 
                                      random_state: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Create k-fold splits at the transcript level, ensuring all replicates stay together.
    
    Args:
        transcript_groups: Dictionary mapping transcript names to file indices
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    # Get list of transcript base names
    transcript_names = list(transcript_groups.keys())
    
    # Create k-fold split on transcript names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_transcript_idx, val_transcript_idx in kf.split(transcript_names):
        # Convert transcript indices back to file indices
        train_file_indices = []
        val_file_indices = []
        
        # Get file indices for training transcripts
        for transcript_idx in train_transcript_idx:
            transcript_name = transcript_names[transcript_idx]
            train_file_indices.extend(transcript_groups[transcript_name])
        
        # Get file indices for validation transcripts
        for transcript_idx in val_transcript_idx:
            transcript_name = transcript_names[transcript_idx]
            val_file_indices.extend(transcript_groups[transcript_name])
        
        splits.append((train_file_indices, val_file_indices))
    
    return splits

def run_kfold_training(config, model_name: str, plot_results: bool = True, positional_config = None) -> pd.DataFrame:
    """Run K-fold cross validation training with transcript-level splitting."""
    distributed, rank, local_rank, world_size, device = _distributed_context()
    if _is_rank0():
        print(f"Starting K-fold training for {model_name}")
        print(config.input_path, positional_config)
        if distributed:
            print(f"Distributed training: world_size={world_size}")
    if "NTv3" in config.model:
        dataset = NTv3Dataset(input_dir=config.input_path,
                              sequence_length=config.context_length,
                              target = config.target,
                              pretrained_model_name=config.pretrained_model_name)
    else:
        dataset = ORFDataset(input_dir = config.input_path, 
                             positional_config=positional_config,
                             target=config.target)
    model_dir = os.path.join(config.result_path, 'models', model_name)
    save_every_epochs = _int_env("SAVE_EVERY_EPOCHS", 1)
    save_every_steps = _int_env("SAVE_EVERY_STEPS", 0)
    max_train_steps = _int_env("MAX_TRAIN_STEPS", 0)
    max_val_steps = _int_env("MAX_VAL_STEPS", 0)
    skip_final_eval = os.environ.get("SKIP_FINAL_EVAL", "0") == "1"
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "").strip()
    if _is_rank0():
        os.makedirs(model_dir, exist_ok=True)
        print(
            f"Checkpointing: save_every_epochs={save_every_epochs} "
            f"save_every_steps={save_every_steps} "
            f"resume_checkpoint={resume_checkpoint or 'none'} "
            f"max_train_steps={max_train_steps} max_val_steps={max_val_steps} "
            f"skip_final_eval={skip_final_eval}",
            flush=True,
        )
    # Keep dataset indices and transcript-level folds deterministic across
    # filesystems and equivalent input directories.
    file_list = sorted(os.listdir(config.input_path))
    transcript_groups = group_files_by_transcript(file_list)
    if _is_rank0():
        validate_transcript_groups(transcript_groups, expected_replicates=4)
    splits = create_transcript_stratified_splits(transcript_groups, config.n_splits, config.seed)
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        if (fold + 1) > config.stop_splits_at:
            break 
        if _is_rank0():
            _save_fold_manifest(
                config.result_path,
                fold + 1,
                config.seed,
                file_list,
                train_idx,
                val_idx,
            )
        print(f'Fold {fold + 1}/{config.n_splits}')
        print(f'  Training on {len(train_idx)} files from {len(set(file_list[i][:-4].split("_")[0] for i in train_idx))} transcripts')
        print(f'  Validating on {len(val_idx)} files from {len(set(file_list[i][:-4].split("_")[0] for i in val_idx))} transcripts')
        
        train_loader, val_loader = create_data_loaders(
            dataset, train_idx, val_idx, config.num_layers,
            batch_size=config.batch_size, shuffle=True,
            distributed=distributed, rank=rank, world_size=world_size,
        )
        
        model = get_model(model_name, config).to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        criterion = _build_criterion(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                   weight_decay=getattr(config, 'weight_decay', 0.0))
        unwrapped_model = model.module if distributed else model
        effective_train_steps = min(max_train_steps, len(train_loader)) if max_train_steps > 0 else len(train_loader)
        effective_val_steps = min(max_val_steps, len(val_loader)) if max_val_steps > 0 else len(val_loader)
        scheduler, scheduler_info = _build_lr_scheduler(
            optimizer,
            config.num_epochs,
            effective_train_steps,
        )
        if _is_rank0():
            print(
                "LR scheduler: "
                f"name={scheduler_info['name']} "
                f"total_updates={scheduler_info['total_updates']} "
                f"warmup_steps={scheduler_info['warmup_steps']} "
                f"min_lr={scheduler_info['min_lr']} "
                f"warmup_lr_init={scheduler_info['warmup_lr_init']}",
                flush=True,
            )
        loss_history = []
        global_step = 0
        start_epoch = 0
        checkpoint_dir = os.path.join(config.result_path, "checkpoints", model_name, f"fold_{fold + 1}")
        timing_path = os.path.join(
            config.result_path,
            "timings",
            model_name,
            f"fold_{fold + 1}",
            "epoch_timings.jsonl",
        )

        if resume_checkpoint:
            start_epoch, global_step, loss_history = _load_checkpoint(
                resume_checkpoint, unwrapped_model, optimizer, device, scheduler=scheduler
            )
        _barrier()

        epoch_iter = tqdm(range(start_epoch, config.num_epochs), bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") if _is_rank0() else range(start_epoch, config.num_epochs)
        for epoch in epoch_iter:
            _barrier()
            _synchronize_device(device)
            epoch_started = time.perf_counter()

            if distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            def save_step_checkpoint(step):
                if scheduler is not None:
                    scheduler.step_update(step)
                if _is_rank0() and save_every_steps > 0 and step % save_every_steps == 0:
                    step_path = os.path.join(checkpoint_dir, f"step_{step:08d}.pt")
                    _save_checkpoint(step_path, unwrapped_model, optimizer, loss_history, fold + 1, epoch + 1, step, config, scheduler=scheduler)

            train_started = time.perf_counter()
            train_loss, train_loss_dict, global_step = train_epoch(
                config, model_name, model, train_loader, criterion, optimizer, device,
                global_step=global_step,
                step_callback=save_step_checkpoint,
                max_steps=max_train_steps,
            )
            _synchronize_device(device)
            _barrier()
            train_seconds = time.perf_counter() - train_started

            val_started = time.perf_counter()
            val_loss, val_loss_dict = validate_epoch(
                config, model_name, model, val_loader, criterion, device,
                max_steps=max_val_steps,
            )
            _synchronize_device(device)
            _barrier()
            val_seconds = time.perf_counter() - val_started
            def filter_dict(d):
                return {k: v for k, v in d.items() if k.endswith("_weighted") or k == "total"}

            loss_history.append({
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in filter_dict(train_loss_dict).items()},
                **{f"val_{k}":   v for k, v in filter_dict(val_loss_dict).items()},
            })

            checkpoint_started = time.perf_counter()
            checkpoint_saved = save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0
            if _is_rank0() and checkpoint_saved:
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}.pt")
                _save_checkpoint(epoch_path, unwrapped_model, optimizer, loss_history, fold + 1, epoch + 1, global_step, config, scheduler=scheduler)
            _barrier()
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            epoch_seconds = time.perf_counter() - epoch_started

            if _is_rank0():
                train_seconds_per_step = train_seconds / effective_train_steps if effective_train_steps else None
                val_seconds_per_step = val_seconds / effective_val_steps if effective_val_steps else None
                timing_record = {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "world_size": world_size,
                    "gpu_batch_size": config.batch_size,
                    "global_batch_size": config.batch_size * world_size,
                    "train_steps": effective_train_steps,
                    "val_steps_per_rank": effective_val_steps,
                    "train_seconds": train_seconds,
                    "val_seconds": val_seconds,
                    "checkpoint_saved": checkpoint_saved,
                    "checkpoint_seconds": checkpoint_seconds,
                    "epoch_seconds": epoch_seconds,
                    "train_seconds_per_step": train_seconds_per_step,
                    "val_seconds_per_step": val_seconds_per_step,
                }
                _append_epoch_timing(timing_path, timing_record)
                train_rate = f"{train_seconds_per_step:.3f}s/step" if train_seconds_per_step is not None else "n/a"
                val_rate = f"{val_seconds_per_step:.3f}s/step" if val_seconds_per_step is not None else "n/a"
                print(
                    "[timing] "
                    f"fold={fold + 1} epoch={epoch + 1} "
                    f"train={train_seconds:.2f}s/{effective_train_steps}steps "
                    f"({train_rate}) "
                    f"val={val_seconds:.2f}s/{effective_val_steps}steps "
                    f"({val_rate}) "
                    f"checkpoint={checkpoint_seconds:.2f}s saved={checkpoint_saved} "
                    f"total={epoch_seconds:.2f}s "
                    f"file={timing_path}",
                    flush=True,
                )

        # Optional: print breakdown every N epochs
        if _is_rank0() and loss_history and (loss_history[-1]["epoch"]) % 10 == 0:
            last_epoch_losses = loss_history[-1]
            loss_names = [
                key.removeprefix("train_")
                for key in last_epoch_losses
                if key.startswith("train_")
            ]
            breakdown = "  |  ".join(
                f"{name}: T={last_epoch_losses.get(f'train_{name}', 0):.4f} "
                f"V={last_epoch_losses.get(f'val_{name}', 0):.4f}"
                for name in loss_names
            )
            print(f"Epoch {loss_history[-1]['epoch']} — {breakdown}")

        if _is_rank0() and plot_results and not skip_final_eval:
            plot_loss_dir = os.path.join(config.result_path, 'loss_curve', f'{model_name}_fold_{fold + 1}')
            os.makedirs(plot_loss_dir, exist_ok=True)
            zero_loss = compute_zero_output_loss(config, val_loader, criterion, device)
            plot_loss_history(loss_history, fold + 1, zero_loss, save_path=plot_loss_dir)
        
        if not _is_rank0():
            _barrier()
            continue

        if skip_final_eval:
            model_path = os.path.join(model_dir, f'model_fold{fold + 1}_smoke.pt')
            _save_checkpoint(model_path, unwrapped_model, optimizer, loss_history, fold + 1, loss_history[-1]["epoch"], global_step, config, scheduler=scheduler)
            results.append({
                'fold': fold + 1,
                'num_parameters': count_parameters(unwrapped_model),
                'train_pearson_corr': np.nan,
                'train_auc': np.nan,
                'train_pr_auc': np.nan,
                'val_pearson_corr': np.nan,
                'val_auc': np.nan,
                'val_pr_auc': np.nan,
                'val_pos_rate': np.nan,
                'final_train_loss': loss_history[-1]['train_total'],
                'final_val_loss':   loss_history[-1]['val_total'],
                'train_files': len(train_idx),
                'val_files': len(val_idx),
                'train_transcripts': len(set(file_list[i][:-4].split("_")[0] for i in train_idx)),
                'val_transcripts': len(set(file_list[i][:-4].split("_")[0] for i in val_idx))
            })
            _barrier()
            continue

        train_metrics = evaluate_model(config, unwrapped_model, train_loader, device)
        val_metrics = evaluate_model(config, unwrapped_model, val_loader, device)
        
        val_predictions_dir = os.path.join(config.result_path, 'validation_predictions', model_name, f'fold_{fold + 1}')
        os.makedirs(val_predictions_dir, exist_ok=True)
        
        val_transcript_names = [file_list[idx][:-4] for idx in val_idx]
        save_validation_predictions(config, unwrapped_model, val_loader, device, val_idx, val_transcript_names, 
                                  config.input_path, val_predictions_dir, fold + 1, model_name)
        
        results.append({
            'fold': fold + 1,
            'num_parameters': count_parameters(model),
            'train_pearson_corr': train_metrics['pearson_corr'],
            'train_auc': train_metrics['auc'],
            'train_pr_auc': train_metrics.get('pr_auc', 0.0),
            'val_pearson_corr': val_metrics['pearson_corr'],
            'val_auc': val_metrics['auc'],
            'val_pr_auc': val_metrics.get('pr_auc', 0.0),
            'val_pos_rate': val_metrics.get('pos_rate', 0.0),
            'final_train_loss': loss_history[-1]['train_total'],
            'final_val_loss':   loss_history[-1]['val_total'],
            'train_files': len(train_idx),
            'val_files': len(val_idx),
            'train_transcripts': len(set(file_list[i][:-4].split("_")[0] for i in train_idx)),
            'val_transcripts': len(set(file_list[i][:-4].split("_")[0] for i in val_idx))
        })

        model_path = os.path.join(model_dir, f'model_fold{fold + 1}.pt')
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss_history': loss_history,
            'fold': fold + 1,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'config': {
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs
            }
        }, model_path)
        print(f"Saved model weights for fold {fold + 1} to: {model_path}")
        _barrier()

    if not _is_rank0():
        return pd.DataFrame(results)

    all_val_predictions_dir = os.path.join(config.result_path, 'validation_predictions', model_name)
    combined_val_file = os.path.join(config.result_path, f'{model_name}_combined_validation_predictions.csv')
    
    if os.path.exists(all_val_predictions_dir):
        combine_validation_predictions(all_val_predictions_dir, combined_val_file)
        print(f"Combined validation predictions saved to: {combined_val_file}")

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(config.result_path, f'{model_name}_kfold_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"Validation AUC: {results_df['val_auc'].mean():.4f} ± {results_df['val_auc'].std():.4f}")
    print(f"Validation Correlation: {results_df['val_pearson_corr'].mean():.4f} ± {results_df['val_pearson_corr'].std():.4f}")
    
    return results_df

def save_validation_predictions(config, model, val_loader, device, val_indices, transcript_names, 
                              input_dir, output_dir, fold_num, model_name):
    model.eval()
    
    has_attention = 'Attention' in model_name or 'Attn' in model_name or hasattr(model, 'get_attention_weights')
    is_unet_model = 'UNet' in model_name or 'Attn' in model_name

    from dataset import ORFDataset
    codon_vocab = ORFDataset._get_codons()
    
    with torch.no_grad():
        batch_idx = 0
        for i, (batch) in enumerate(val_loader):
            if "NTv3" in config.model:
                batch_input, batch_lengths = batch
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_input.items()
                }
                batch_lengths = batch_lengths.to(device)
                mask = create_mask_from_lengths(batch_lengths, batch_on_device["tokens"].shape[1], device)
                all_outputs = model(tokens=batch_on_device["tokens"], conditions=batch_on_device["extra_features"])
            else:
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch_lengths = batch_on_device["lengths"]
                mask = create_mask_from_lengths(batch_lengths, batch_on_device["tokens"].shape[1], device)
                all_outputs = model(batch_on_device["tokens"], mask)

            regression_out      = all_outputs["regression"]                          # [B, L]
            codon_logits        = all_outputs["classification"]                      # [B, L, 65]
            predicted_codon_ids = codon_logits.argmax(dim=-1)                        # [B, L]
            binalization_out    = torch.sigmoid(all_outputs["binarization"])         # [B, L]

            attention_weights = None
            if has_attention and hasattr(model, 'get_attention_weights'):
                try:
                    attention_weights = model.get_attention_weights()
                except Exception as e:
                    print(f"Warning: Could not extract attention weights: {e}")
            
            for j in range(regression_out.shape[0]):
                sample_idx = batch_idx * val_loader.batch_size + j
                if sample_idx >= len(transcript_names):
                    continue

                transcript_name  = transcript_names[sample_idx]
                actual_length    = batch_lengths[j].item()
                prediction       = regression_out[j, :actual_length].cpu().numpy().squeeze()
                codon_ids        = predicted_codon_ids[j, :actual_length].cpu().numpy()
                predicted_codons = [codon_vocab[idx] for idx in codon_ids]
                binalized        = binalization_out[j, :actual_length].cpu().numpy().squeeze()

                attention_data = {}
                if attention_weights:
                    attention_data = process_attention_for_sample(
                        attention_weights, j, actual_length,
                        batch_on_device["tokens"].shape[1], is_unet_model
                    )

                original_file_path = os.path.join(input_dir, f"{transcript_name}.txt")
                if os.path.exists(original_file_path):
                    try:
                        df = pd.read_csv(original_file_path, sep="\t")
                        if df.shape[1] == 1:
                            df = pd.read_csv(original_file_path, sep=",")
                        df_copy = df.copy()

                        min_len = min(len(prediction), len(df_copy))
                        df_copy.loc[:min_len-1, 'prediction']          = prediction[:min_len]
                        df_copy.loc[:min_len-1, 'predicted_codon']     = predicted_codons[:min_len]
                        df_copy.loc[:min_len-1, 'predicted_binalized'] = binalized[:min_len]

                        for att_col, att_values in attention_data.items():
                            att_min = min(len(att_values), len(df_copy))
                            df_copy.loc[:att_min-1, att_col] = att_values[:att_min]

                        df_copy['model']           = f"{model_name}_fold_{fold_num}"
                        df_copy['transcript_name'] = transcript_name
                        output_file_path = os.path.join(output_dir, f"{transcript_name}.txt")
                        df_copy.to_csv(output_file_path, sep=",", index=False)

                    except Exception as e:
                        print(f"Error processing {transcript_name}: {e}")

            batch_idx += 1

def process_attention_for_sample(attention_weights, sample_idx, original_length, padded_length, is_unet_model):

    attention_data = {}
    
    if is_unet_model:
        attention_data = process_attention_weights_unet_single(
            attention_weights, sample_idx, original_length, padded_length
        )
    else:
        # Process attention for non-UNet models
        for level, weights in attention_weights.items():
            # Extract weights for the specific sample
            if len(weights.shape) > 2:
                if weights.shape[0] > sample_idx:
                    att_weights = weights[sample_idx, 0] if weights.shape[1] == 1 else weights[sample_idx]
                else:
                    att_weights = weights[0, 0] if weights.shape[1] == 1 else weights[0]
            elif len(weights.shape) == 2:
                att_weights = weights[sample_idx] if weights.shape[0] > sample_idx else weights[0]
            else:
                att_weights = weights
            
            # Convert to numpy if it's a tensor
            if hasattr(att_weights, 'cpu'):
                att_weights = att_weights.cpu().numpy()
            
            # Resize to match original length
            if len(att_weights) > original_length:
                att_weights = att_weights[:original_length]
            elif len(att_weights) < original_length:
                att_weights = np.pad(att_weights, (0, original_length - len(att_weights)), 'edge')
            
            attention_data[f'attention_{level}'] = att_weights
    
    return attention_data

def process_attention_weights_unet_single(attention_weights, sample_idx, original_length, padded_length):
    """Process attention weights for UNet models for a single sample."""
    processed_weights = {}
    for level, weights in attention_weights.items():
        # Extract weights for the specific sample
        if len(weights.shape) > 2:
            if weights.shape[0] > sample_idx:
                att_weights = weights[sample_idx, 0] if weights.shape[1] == 1 else weights[sample_idx]
            else:
                att_weights = weights[0, 0] if weights.shape[1] == 1 else weights[0]
        elif len(weights.shape) == 2:
            att_weights = weights[sample_idx] if weights.shape[0] > sample_idx else weights[0]
        else:
            att_weights = weights
        if hasattr(att_weights, 'cpu'):
            att_weights = att_weights.cpu().numpy()
        
        factor = padded_length // len(att_weights)
        original_length_at_level = original_length // factor
        att_weights_cropped = att_weights[:original_length_at_level]
        
        if factor > 1:
            att_weights_upsampled = np.repeat(att_weights_cropped, factor)
            if len(att_weights_upsampled) > original_length:
                att_weights_upsampled = att_weights_upsampled[:original_length]
            elif len(att_weights_upsampled) < original_length:
                padding_needed = original_length - len(att_weights_upsampled)
                att_weights_upsampled = np.pad(att_weights_upsampled, (0, padding_needed), 'edge')
        else:
            att_weights_upsampled = att_weights_cropped[:original_length]
        
        processed_weights[f'attention_{level}'] = att_weights_upsampled
    
    return processed_weights

def combine_validation_predictions(input_dir, output_file):
    """Combine validation predictions from all folds into a single file."""
    txt_files = []
    
    # Collect all .txt files from all fold subdirectories
    for fold_dir in Path(input_dir).iterdir():
        if fold_dir.is_dir():
            txt_files.extend(fold_dir.glob("*.txt"))
    
    if not txt_files:
        print("No validation prediction files found to combine")
        return
    
    combined_data = []
    
    for file_path in tqdm(sorted(txt_files), desc="Combining validation predictions"):
        try:
            df = pd.read_csv(file_path, sep='\t')
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, sep=',')
            
            # Ensure transcript_name is present
            if 'transcript_name' not in df.columns:
                df.insert(0, 'transcript_name', file_path.stem)
            
            combined_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if combined_data:
        print('Concatenating validation predictions...')
        final_df = pd.concat(combined_data, ignore_index=True)
        print('Concatenation done, saving...')
        final_df.to_csv(output_file, index=False)
        print('Save completed')
        
        return final_df
    else:
        print("No valid validation prediction files found")
        return None

def train_multiple_models(config, model_names: List[str], plot_results: bool = True) -> Dict[str, pd.DataFrame]:
    """Train multiple models using K-fold cross validation."""
    all_results = {}
    
    for model_name in model_names:
        print(f"\nTraining model: {model_name}")
        try:
            results_df = run_kfold_training(config, model_name, plot_results)
            all_results[model_name] = results_df
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    if all_results:
        combined_results = []
        for model_name, results_df in all_results.items():
            results_with_model = results_df.copy()
            results_with_model['model_name'] = model_name
            combined_results.append(results_with_model)
        
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_csv_path = os.path.join(config.result_path, 'combined_kfold_results.csv')
        combined_df.to_csv(combined_csv_path, index=False)
    
    return all_results

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model being trained
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
