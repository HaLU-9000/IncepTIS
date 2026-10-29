"""
Dataset classes and data loading utilities for the ORF prediction project.
"""

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial 
import multiprocessing as mp
from typing import Tuple, List, Optional
from transformers import AutoTokenizer

from utils import PositionalEncodingConfig


class ORFDataset(Dataset):
    """Dataset for ORF prediction from DNA sequences with optional positional encoding."""
    
    def __init__(
        self,
        input_dir: str,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
        positional_config: Optional[PositionalEncodingConfig] = None,
        target: str = "normalized_similarity",
    ):
        
        self.file_list = []  # ← add this
        self.positional_config = positional_config
        self.samples = []
        self.target = target

        self.codon_vocab = self._get_codons()
        self.codon_stoi = {codon: i for i, codon in enumerate(self.codon_vocab)}
        self.num_codons = len(self.codon_vocab)   # 65

        file_paths = [
            os.path.join(input_dir, filename)
            for filename in sorted(os.listdir(input_dir))
        ]
        for file_path in tqdm(
            file_paths,
            desc="Loading dataset",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        ):
            result = self._process_file_with_config(self.positional_config, file_path)
            if result is None or result["tokens"] is None:
                print(f"Skipped: {file_path}") 
                continue
            self.samples.append(result)

    def _process_file_with_config(
        self,
        positional_config,
        file_path: str,
    ):
        df = pd.read_csv(file_path, sep=",")

        if df.empty:
            return None

        seq_length = len(df)
        base_seq = ORFDataset._convert_bases_vectorized(df["base"].values)
        condition_feature = ORFDataset._get_condition_feature(df)
        feature_list = [base_seq]

        if condition_feature is not None:
            condition_column = np.full((seq_length, 1), condition_feature, dtype=np.float32)
            feature_list.append(condition_column)
        else:
            condition_column = np.zeros((seq_length, 1), dtype=np.float32)
            feature_list.append(condition_column)

        if positional_config is not None:
            positional_encoding = positional_config.create_encoding(seq_length)
            feature_list.append(positional_encoding)

        input_features = np.concatenate(feature_list, axis=1)
        sim_seq = df[self.target].values.astype(np.float32)

        codon_seq = df["peak_mask"].astype(str).tolist()

        codon_ids = [
            self.codon_stoi[codon] if codon in self.codon_stoi else self.codon_stoi["False"]
            for codon in codon_seq
        ]

        # one-hot 化
        codon_ids_tensor = torch.tensor(codon_ids, dtype=torch.long)
        codon_onehot = F.one_hot(codon_ids_tensor, num_classes=self.num_codons).to(torch.float32)

        input_tensor = torch.from_numpy(input_features).to(torch.float32)
        target_tensor = torch.from_numpy(sim_seq).to(torch.float32)


        return {
            "tokens": input_tensor,
            "targets": target_tensor,
            "codon_seq": codon_onehot,
        }

    def get_feature_info(self) -> dict:
        """Get information about input features."""
        if len(self.samples) == 0:
            return {}

        sample = self.samples[0]
        sample_input = sample["tokens"]
        total_features = sample_input.shape[-1]

        base_features = 4
        condition_features = 1
        positional_features = total_features - base_features - condition_features

        return {
            "total_features": total_features,
            "base_features": base_features,
            "condition_features": condition_features,
            "positional_features": positional_features,
            "positional_type": self.positional_config.encoding_type if self.positional_config else None,
            "positional_dim": self.positional_config.d_model if positional_features > 0 and self.positional_config else 0,
            "codon_vocab_size": self.num_codons,
        }

    @staticmethod
    def _get_condition_feature(df: pd.DataFrame) -> Optional[float]:
        if "condition" in df.columns:
            condition_values = df["condition"].dropna().unique()
            if len(condition_values) > 0:
                condition_value = float(condition_values[0])
                return condition_value
        return 0.0

    @staticmethod
    def _convert_bases_vectorized(bases: np.ndarray) -> np.ndarray:
        output = np.zeros((len(bases), 4), dtype=np.float32)

        mask_A = bases == "A"
        mask_C = bases == "C"
        mask_G = bases == "G"
        mask_T = bases == "T"
        mask_N = ~(mask_A | mask_C | mask_G | mask_T)

        output[mask_A] = [1, 0, 0, 0]
        output[mask_C] = [0, 1, 0, 0]
        output[mask_G] = [0, 0, 1, 0]
        output[mask_T] = [0, 0, 0, 1]
        output[mask_N] = [0.25, 0.25, 0.25, 0.25]

        return output

    @staticmethod
    def _get_codons():
        bases = ["A", "T", "G", "C"]
        codons = ["False"]
        for base0 in bases:
            for base1 in bases:
                for base2 in bases:
                    codons.append(base0 + base1 + base2)
        return codons

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class NTv3Dataset(Dataset):
    """
    Reads CSVs once in __init__, stores ready-to-train tensors.
    Output schema matches GenomeBigWigDataset: tokens + bigwig_targets + coords (+ optional extra_features).
    """

    def __init__(
        self,
        input_dir: str,
        sequence_length: int,
        pretrained_model_name:str,
        transform_fn=None,
        keep_target_center_fraction: float = 1.0,
        positional_config=None,
        include_extra_features: bool = True,
        target:str = "cosine similarity"
    ):
        super().__init__()
        self.samples = []
        self.target_key = target
        self.sequence_length = sequence_length
        self.transform_fn = transform_fn or (lambda x: x)
        self.keep_target_center_fraction = keep_target_center_fraction
        self.positional_config = positional_config
        self.include_extra_features = include_extra_features
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True)
        print("TARGET ", target)

        self.codon_vocab = ORFDataset._get_codons()
        self.codon_stoi = {codon: i for i, codon in enumerate(self.codon_vocab)}
        self.num_codons = len(self.codon_vocab)  # 65

        self.file_paths = [
            os.path.join(input_dir, filename)
            for filename in sorted(os.listdir(input_dir))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        df = pd.read_csv(fp)
        seq = "".join(df["base"].astype(str).values)
        seq_length = len(df)

        tok = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="pt",
        )
        tokens = tok["input_ids"][0]

        y = df[self.target_key].values.astype(np.float32)
        if len(y) >= self.sequence_length:
            y = y[:self.sequence_length]
        else:
            y = np.concatenate([y, np.zeros(self.sequence_length - len(y), np.float32)], axis=0)

        targets = torch.from_numpy(y).float().unsqueeze(-1)  # (L, 1)
        targets = torch.nan_to_num(targets, nan=0.0)
        targets = self.transform_fn(targets)
        
        sample = {
            "tokens" : tokens,
            "targets": targets,
            "start"  : 0,
            "end"    : self.sequence_length,
            "path"   : fp,
            "length" :seq_length
        }

        codon_seq = df["peak_mask"].astype(str).tolist()
        codon_ids = [
            self.codon_stoi[codon] if codon in self.codon_stoi else self.codon_stoi["False"]
            for codon in codon_seq
        ]
        codon_ids_tensor = torch.tensor(codon_ids, dtype=torch.long)
        codon_onehot = F.one_hot(codon_ids_tensor, num_classes=self.num_codons).float()  # (seq_length, 65)
        
        # pad/truncate to sequence_length, same as bigwig_targets
        if len(codon_onehot) >= self.sequence_length:
            codon_onehot = codon_onehot[:self.sequence_length]
        else:
            pad = torch.zeros(self.sequence_length - len(codon_onehot), self.num_codons)
            codon_onehot = torch.cat([codon_onehot, pad], dim=0)  # (sequence_length, 65)
        
        sample["codon_seq"] = codon_onehot
        sample["targets"] = targets.squeeze(-1)  # [L] to match ORFDataset convention
        if self.include_extra_features:
            cond = 0.0
            if "condition" in df.columns:
                vals = df["condition"].dropna().unique()
                if len(vals) > 0:
                    cond = float(vals[0])

            extra = [torch.full((self.sequence_length, 1), cond, dtype=torch.float32)]
            if self.positional_config is not None:
                pos = self.positional_config.create_encoding(self.sequence_length)  # numpy (L, d)
                extra.append(torch.from_numpy(pos).float())
            extra_features = torch.cat(extra, dim=1)
            sample["extra_features"] = extra_features

        return sample


class Collate_Fn_Variable_Length:
    """
    Custom collate function to handle variable length sequences using padding.
    
    Args:
        batch: List of (input_tensor, target_tensor) tuples
        
    Returns:
        Tuple of (padded_inputs, padded_targets, lengths)
    """
    def __init__(self, pad_size):
        self.n = pad_size
    def __call__(self, batch):
        inputs = [item["tokens"] for item in batch]
        targets = [item["targets"] for item in batch]
        codon_seqs = [item["codon_seq"] for item in batch]

        lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
        max_length = lengths.max().item()
        n = self.n

        padded_length = ((max_length + 2 ** n - 1) // (2 ** n)) * (2 ** n)
        batch_size = len(inputs)
        feature_dim = inputs[0].shape[-1]
        codon_dim = codon_seqs[0].shape[-1]   # 65

        padded_inputs = torch.zeros(batch_size, padded_length, feature_dim, dtype=torch.float32)
        padded_targets = torch.full((batch_size, padded_length), -1.0, dtype=torch.float32)
        padded_codons = torch.zeros(batch_size, padded_length, codon_dim, dtype=torch.float32)

        for i, (input_seq, target_seq, codon_seq) in enumerate(zip(inputs, targets, codon_seqs)):
            seq_len = len(input_seq)
            padded_inputs[i, :seq_len] = input_seq
            padded_targets[i, :seq_len] = target_seq
            padded_codons[i, :seq_len] = codon_seq

        return {
            "tokens": padded_inputs,
            "targets": padded_targets,
            "lengths": lengths,
            "codon_seq": padded_codons,
        }
    
class CollateNTv3WithLengths:
    """
    Collate function for NTv3Dataset.
    Returns:
        batch (dict of stacked tensors)
        lengths (LongTensor)
    """

    def __call__(self, batch):
        # batch: List[Dict]
        lengths = torch.tensor(
            [sample["length"] for sample in batch],
            dtype=torch.long
        )

        collated = {}

        # Stack tensor fields
        for key in batch[0].keys():
            value = batch[0][key]

            if torch.is_tensor(value):
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )
            else:
                # keep non-tensors as list (e.g. path)
                collated[key] = [sample[key] for sample in batch]

        return collated, lengths

def create_mask_from_lengths(
        lengths: torch.Tensor, 
        max_length: Optional[int] = None, 
        device: Optional[torch.device] = None, 
        do_unsqueeze=False) -> torch.Tensor:
    """
    Create binary mask from sequence lengths.
    
    Args:
        lengths: Tensor of sequence lengths
        max_length: Maximum length for the mask
        device: Device to place the mask on
        
    Returns:
        Binary mask tensor
    """
    if max_length is None:
        max_length = lengths.max().item()
    if device is None:
        device = lengths.device
    
    batch_size = lengths.size(0)
    mask = torch.arange(max_length, device=device).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    if do_unsqueeze:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
    return mask

def create_data_loaders(dataset: ORFDataset, train_indices: List[int], val_indices: List[int], pad_size,
                       batch_size: int = 32, shuffle: bool = True,
                       distributed: bool = False,
                       rank: int = 0,
                       world_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        dataset: The dataset to split
        train_indices: Indices for training set
        val_indices: Indices for validation set
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import Subset
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    if dataset.__class__ == NTv3Dataset:
        collate_fn_variable_length = CollateNTv3WithLengths()
    else:
        collate_fn_variable_length = Collate_Fn_Variable_Length(pad_size=pad_size)
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        shuffle = False

    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size,
        collate_fn=collate_fn_variable_length, 
        shuffle=shuffle,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_variable_length, 
        shuffle=False
    )
    
    return train_loader, val_loader
