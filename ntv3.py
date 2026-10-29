from typing import Callable, Dict, List

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM


def _hf_token():
    """Return the only supported source of Hugging Face credentials."""
    return os.environ.get("HF_TOKEN")

def crop_center(x: np.ndarray, keep_target_center_fraction: float = 0.375) -> np.ndarray:
    """Crop the central sequence-length fraction for arrays of size (..., seq_len, num_tracks)"""
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]

class LinearHead(nn.Module):
    """A linear head that predicts one scalar value per track."""
    def __init__(self, embed_dim: int, num_labels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.head(x)
        return x

class HFModelWithHead(nn.Module):
    """Simple model wrapper: HF backbone + bigwig head."""
    
    def __init__(self, config):
        super().__init__()
        model_name = config.pretrained_model_name
        self.ntv3_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, token=_hf_token())
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.ntv3_config, token=_hf_token(),
        )
        _lora_rank = getattr(config, "lora_rank", 0)
        if _lora_rank and _lora_rank > 0:
            from ntv3_dilated_heads import _apply_lora_to_backbone
            backbone = _apply_lora_to_backbone(
                backbone, _lora_rank,
                getattr(config, "lora_alpha", 16),
                getattr(config, "lora_dropout", 0.05))
            print(f"[ntv3] Backbone + LoRA (rank={_lora_rank})")
            self.backbone = backbone  # skip torch.compile for peft compatibility
        else:
            if getattr(config, "freeze_backbone", False):
                for _p in backbone.parameters():
                    _p.requires_grad_(False)
                _n = sum(_p.numel() for _p in backbone.parameters())
                print(f"[ntv3] Backbone FROZEN ({_n/1e6:.1f}M params)")
            print("COMPILING MODEL...")
            self.backbone = torch.compile(backbone)
        
        embed_dim = self.ntv3_config.embed_dim
        
        # Bigwig head (NTv3 outputs at single-nucleotide resolution)
        self.head = LinearHead(embed_dim + 1, 1)
        self.model_name = model_name
    
    def forward(self, tokens: torch.Tensor, conditions: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Forward through backbone
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]  # Last hidden state
        bigwig_logits = self.head(torch.cat([embedding, conditions], dim=2))
        
        return {"prediction": bigwig_logits}
    
class HFModelWithMLP(nn.Module):
    """Simple model wrapper: HF backbone + bigwig head."""
    
    def __init__(self, config):
        super().__init__()
        model_name = config.pretrained_model_name
        self.ntv3_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, token=_hf_token())
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.ntv3_config, token=_hf_token(),
        )
        _lora_rank = getattr(config, "lora_rank", 0)
        if _lora_rank and _lora_rank > 0:
            from ntv3_dilated_heads import _apply_lora_to_backbone
            backbone = _apply_lora_to_backbone(
                backbone, _lora_rank,
                getattr(config, "lora_alpha", 16),
                getattr(config, "lora_dropout", 0.05))
            print(f"[ntv3] Backbone + LoRA (rank={_lora_rank})")
            self.backbone = backbone  # skip torch.compile for peft compatibility
        else:
            if getattr(config, "freeze_backbone", False):
                for _p in backbone.parameters():
                    _p.requires_grad_(False)
                _n = sum(_p.numel() for _p in backbone.parameters())
                print(f"[ntv3] Backbone FROZEN ({_n/1e6:.1f}M params)")
            print("COMPILING MODEL...")
            self.backbone = torch.compile(backbone)
        
        embed_dim = self.ntv3_config.embed_dim
        
        in_dim = embed_dim + 1  # +1 for condition

        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm(in_dim),
        )

        # regression head: scalar per position
        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

        # classification head: 65-class codon logits per position
        self.classification_head = nn.Linear(in_dim, 65)

        self.model_name = model_name

        self.logit_threshold  = nn.Parameter(torch.logit(
                                    torch.tensor(config.init_threshold)))
    
    def forward(self, tokens: torch.Tensor, conditions: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]                      
        x = torch.cat([embedding, conditions], dim=-1)               
        x = self.shared_mlp(x)                                       

        output         = self.regression_head(x)
        binarization   = output - self.logit_threshold
        classification = self.classification_head(x)

        return {
            "regression":     output.squeeze(-1),
            "classification": classification,
            "binarization":   binarization.squeeze(-1),
        }

class ResidualConvBlock(nn.Module):
    """Conv1d -> LayerNorm -> ReLU -> Conv1d -> LayerNorm + skip connection"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)          # [B, C, L] for Conv1d
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(1, 2)          # [B, L, C]
        x = self.norm2(x + residual)
        return x


class HFModelWithCNN(nn.Module):
    """HF backbone + ResNet-style CNN + shared heads."""

    def __init__(self, config):
        super().__init__()
        model_name = config.pretrained_model_name
        self.ntv3_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, token=_hf_token())
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=self.ntv3_config, token=_hf_token(),
        )
        _lora_rank = getattr(config, "lora_rank", 0)
        if _lora_rank and _lora_rank > 0:
            from ntv3_dilated_heads import _apply_lora_to_backbone
            backbone = _apply_lora_to_backbone(
                backbone, _lora_rank,
                getattr(config, "lora_alpha", 16),
                getattr(config, "lora_dropout", 0.05))
            print(f"[ntv3] Backbone + LoRA (rank={_lora_rank})")
            self.backbone = backbone  # skip torch.compile for peft compatibility
        else:
            if getattr(config, "freeze_backbone", False):
                for _p in backbone.parameters():
                    _p.requires_grad_(False)
                _n = sum(_p.numel() for _p in backbone.parameters())
                print(f"[ntv3] Backbone FROZEN ({_n/1e6:.1f}M params)")
            print("COMPILING MODEL...")
            self.backbone = torch.compile(backbone)

        embed_dim = self.ntv3_config.embed_dim
        cnn_in_dim = embed_dim + 1  # +1 for condition

        # ResNet-style CNN layers (operate in embedding space before MLP)
        num_cnn_layers = getattr(config, "num_cnn_layers", 4)
        kernel_size    = getattr(config, "cnn_kernel_size", 3)
        self.cnn_layers = nn.ModuleList([
            ResidualConvBlock(cnn_in_dim, kernel_size=kernel_size)
            for _ in range(num_cnn_layers)
        ])


        in_dim = cnn_in_dim 

        self.regression_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )
        self.classification_head = nn.Linear(in_dim, 65)

        self.model_name = model_name
        self.logit_threshold = nn.Parameter(torch.logit(
                                   torch.tensor(config.init_threshold)))

    def forward(self, tokens: torch.Tensor, conditions: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]               # [B, L, embed_dim]

        x = torch.cat([embedding, conditions], dim=-1)      # [B, L, embed_dim+1]

        for cnn_block in self.cnn_layers:                   # ResNet CNN
            x = cnn_block(x)

        output         = self.regression_head(x)
        binarization   = output - self.logit_threshold
        classification = self.classification_head(x)

        return {
            "regression":     output.squeeze(-1),
            "classification": classification,
            "binarization":   binarization.squeeze(-1),
        }                


if __name__ == "__main__":
    HF_TOKEN = os.getenv("HF_TOKEN", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Choose dtype (bf16 if supported; else fp16 on GPU; else fp32)
    if device == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        torch_dtype = torch.float32

    print("torch_dtype:", torch_dtype)


    model_name = "InstaDeepAI/NTv3_650M_pre"

    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Tokenize input sequences
    batch = tok(["ATCGNATCG", "ACGT"], add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")

    # Run model
    out = model(**batch)

    # Print output shapes
    print(out.logits.shape) 
