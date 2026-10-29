"""
Configuration file for the ORF prediction project.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from utils import PositionalEncodingConfig
import json

@dataclass
class LossHeadConfig:
    weight: float = 1.0
    kind  : str   = "mse"
    # regression
    huber_delta: float = 1.0
    # binarization focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # classification focal
    gamma: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "LossHeadConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

@dataclass
class LossConfig:
    regression    : LossHeadConfig = field(default_factory=lambda: LossHeadConfig(weight=1.0, kind="mse"))
    classification: LossHeadConfig = field(default_factory=lambda: LossHeadConfig(weight=0.3, kind="multiclass_focal"))
    binarization  : LossHeadConfig = field(default_factory=lambda: LossHeadConfig(weight=0.2, kind="focal"))

    @classmethod
    def from_dict(cls, d: dict) -> "LossConfig":
        return cls(
            regression     = LossHeadConfig.from_dict(d.get("regression",     {})),
            classification = LossHeadConfig.from_dict(d.get("classification", {})),
            binarization   = LossHeadConfig.from_dict(d.get("binarization",   {})),
        )
    

@dataclass
class Config:
    # Data paths
    base_dir  : str = "/home/haruhiko/Documents/PredicTIS"
    input_dir : str = "input_data"
    result_dir: str = "results"
    target    : str = "cosine similarity"
    
    # Model parameters
    input_features  : int   = 5
    num_features    : int   = 64
    dropout_prob    : float = 0.1
    num_layers      : int   = 3
    attention_layers        = [1,2,3]
    init_threshold  : float = 0.2
    
    # transformers parameters
    nhead = 8
    batch_first = True

    # Hyena parameters
    d_model:int = num_features
    context_length:int = 65536 # 2**16
    order:Optional[int] = 4
    short_conv_size:int = 3
    d_filter_mlp= 16
    n_filter_layers = 3
    d_embed = 3
    omega = 8
    use_mlp = False 
    # use_mlp: Hyenafilter setting. 
    #  True: slow, use more memory but param efficient. False: fast, less memory but params become so large.

    #ntv3
    pretrained_model_name = None

    loss: LossConfig = field(default_factory=LossConfig)

    # Training parameters
    seed: int = 524
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    n_splits: int = 10
    pdrop_hyena = dropout_prob

    stop_splits_at : int   = 1     
    weight_decay   : float = 0.0    
    model          : str   = "AttentionUNet1D"
    

    # Transcript grouping settings
    expected_replicates_per_transcript = 4
    transcript_name_pattern = 'underscore_split'  # or 'regex_pattern'
    custom_transcript_regex = None  # Set to custom regex if needed

    use_positional_encoding = False
    positional_encoding_type = 'sinusoidal'  # 'none', 'linear', 'sinusoidal', 'relative', 'codon'
    positional_encoding_dim = 2
    positional_normalize = False
    
    def get_positional_config(self) -> PositionalEncodingConfig:
        """Get positional encoding configuration."""
        encoding_type = self.positional_encoding_type if self.use_positional_encoding else 'none'
        
        return PositionalEncodingConfig(
            encoding_type=encoding_type,
            d_model=self.positional_encoding_dim,
            normalize=self.positional_normalize
        )
    
    def get_total_input_features(self) -> int:
        """Calculate total number of input features."""
        base_features = 4  # DNA bases
        condition_features = 1  # condition
        positional_features = self.positional_encoding_dim if self.use_positional_encoding else 0
        
        return base_features + condition_features + positional_features
    
    def get_model_params(self):
        """Get parameters for model initialization."""
        return {
            'in_channels': self.get_total_input_features(),
            'num_features': self.get_total_input_features()
        }
    
    def __post_init__(self):
        self.input_path = os.path.join(self.base_dir, self.input_dir)
        self.result_path = os.path.join(self.base_dir, self.result_dir)
        
    def get_model_params(self) -> Dict[str, Any]:
        return {
            'input_features': self.input_features,
            'num_features': self.num_features,
            'dropout_prob': self.dropout_prob,
            'num_layers'  : self.num_layers,
            'attention_layers':self.attention_layers
        }
        
    @classmethod
    def from_json(cls, path: str) -> "Config":
        import json
        with open(path) as f:
            d = json.load(f)

        cfg = cls()

        for k, v in d.get("paths", {}).items():
            setattr(cfg, k, v)

        for k, v in d.get("data", {}).items():
            setattr(cfg, k, v)

        for k, v in d.get("model", {}).items():
            setattr(cfg, k, v)

        for k, v in d.get("hyena", {}).items():
            setattr(cfg, k, v)

        pe = d.get("positional_encoding", {})
        cfg.use_positional_encoding  = pe.get("use",       cfg.use_positional_encoding)
        cfg.positional_encoding_type = pe.get("type",      cfg.positional_encoding_type)
        cfg.positional_encoding_dim  = pe.get("dim",       cfg.positional_encoding_dim)
        cfg.positional_normalize     = pe.get("normalize", cfg.positional_normalize)

        for k, v in d.get("training", {}).items():
            setattr(cfg, k, v)

        cfg.loss = LossConfig.from_dict(d.get("loss", {}))

        cfg.__post_init__()
        return cfg