"""
Model definitions for the ORF prediction project.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Optional, Tuple
from config import Config
import torch.utils.checkpoint as checkpoint
from types import SimpleNamespace

from transformers import AutoConfig, AutoModelForMaskedLM

def _hf_token():
    """Return the only supported source of Hugging Face credentials."""
    return os.environ.get("HF_TOKEN")

from ntv3 import HFModelWithHead, HFModelWithMLP, HFModelWithCNN


class AttentionGate(nn.Module):
    """Attention Gate module for U-Net with mask support."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int, dropout_prob: float = 0.1):
        """
        Initialize attention gate.
        
        Args:
            F_g: Number of feature maps in gating signal
            F_l: Number of feature maps in skip connection
            F_int: Number of feature maps in intermediate layer
            dropout_prob: Dropout probability
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_attention_weights = None
    
    def forward(self, g: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through attention gate.
        
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
            mask: Binary mask for valid positions
            
        Returns:
            Attended feature maps
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.dropout(psi)
        
        if mask is not None:
            psi = psi * mask.unsqueeze(1)
        
        psi = self.psi(psi)
        if mask is not None:
            psi = psi * mask.unsqueeze(1)
        
        attended_x = x * psi
        if mask is not None:
            attended_x = attended_x * mask.unsqueeze(1)
        
        self.last_attention_weights = psi.detach()
        return attended_x


class MaskedConv1dBlock(nn.Module):
    """Convolutional block with mask support."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 1, dropout_prob: float = 0.1):
        """
        Initialize masked conv block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            dropout_prob: Dropout probability
        """
        super(MaskedConv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional masking."""
        x = self.conv(x)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        return x


class AttentionUNet1D(nn.Module):
    """1D U-Net with attention gates for sequence prediction."""
    
    def __init__(self, config):
        """
        Initialize Attention U-Net.
        
        Args:
            input_features: Number of input features
            num_features: Base number of features
            dropout_prob: Dropout probability
            num_layers: Number of encoder/decoder layers
            attention_layers: List of layer indices to apply attention (1-indexed). 
                            If None, applies attention to all layers.
        """
        super(AttentionUNet1D, self).__init__()
        input_features   = config.input_features
        num_features     = config.num_features    
        dropout_prob     = config.dropout_prob    
        num_layers       = config.num_layers       
        attention_layers = config.attention_layers 
        init_threshold   = config.init_threshold
        
        self.num_layers = num_layers
        self.attention_layers = attention_layers if attention_layers is not None else list(range(1, num_layers + 1))
        self.attention_weights = OrderedDict()
        
        # Validate attention_layers
        for layer in self.attention_layers:
            if layer < 1 or layer > num_layers:
                raise ValueError(f"Attention layer {layer} is out of range [1, {num_layers}]")
        
        # Build encoder layers
        self.encoders = nn.ModuleList()
        in_channels = input_features
        for i in range(num_layers):
            out_channels = num_features * (2 ** i)
            kernel_size = 7 if i == 0 else 3  # First layer uses larger kernel
            padding = 3 if i == 0 else 1
            
            encoder = MaskedConv1dBlock(
                in_channels, out_channels, 
                kernel_size=kernel_size, padding=padding, 
                dropout_prob=dropout_prob
            )
            self.encoders.append(encoder)
            in_channels = out_channels
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Bottleneck
        bottleneck_features = num_features * (2 ** num_layers)
        self.bottleneck = MaskedConv1dBlock(
            num_features * (2 ** (num_layers - 1)), 
            bottleneck_features, 
            dropout_prob=dropout_prob
        )
        
        # Build attention gates (only for specified layers)
        self.attention_gates = nn.ModuleDict()
        for layer in self.attention_layers:
            layer_idx = layer - 1  # Convert to 0-indexed
            F_l = num_features * (2 ** layer_idx)  # Features from encoder
            F_g = F_l  # Features from decoder (after upsampling)
            F_int = max(F_l // 2, num_features // 2)  # Intermediate features
            
            self.attention_gates[f'layer_{layer}'] = AttentionGate(
                F_g=F_g, F_l=F_l, F_int=F_int, dropout_prob=dropout_prob
            )
        
        # Build decoder upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = bottleneck_features if i == 0 else num_features * (2 ** (num_layers - i))
            out_channels = num_features * (2 ** (num_layers - i - 1))
            
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.upsample_layers.append(upsample)
        
        # Build decoder blocks
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            channels = num_features * (2 ** (num_layers - i - 1))
            decoder = MaskedConv1dBlock(channels, channels, dropout_prob=dropout_prob)
            self.decoders.append(decoder)
        
        # Final output
        self.final_conv       = nn.Conv1d(num_features, 1 , kernel_size=1)
        self.categorical_conv = nn.Conv1d(num_features, 65, kernel_size=1)
        self.logit_threshold  = nn.Parameter(torch.logit(
                                            torch.tensor(init_threshold)))
    
    def _apply_mask_to_conv(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply mask after convolution operations."""
        if mask is not None:
            return x * mask.unsqueeze(1)
        return x
    
    def _downsample_mask(self, mask: Optional[torch.Tensor], factor: int = 2) -> Optional[torch.Tensor]:
        """Downsample mask for pooling operations."""
        if mask is None:
            return None
        # Use max pooling to preserve valid positions
        return F.max_pool1d(mask.float().unsqueeze(1), kernel_size=factor).squeeze(1) > 0
    
    def postprocess(self, output, mask):
        output = output.permute(0, 2, 1)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        if mask is not None:
            if output.ndim == 2:
                output = output * mask.float()
            elif output.ndim == 3:
                output = output * mask.unsqueeze(-1).float()
    
        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            mask: Optional mask tensor [batch_size, seq_len]
            
        Returns:
            Output predictions [batch_size, seq_len]
        """
        # Permute for convolution: [B, seq_len, features] -> [B, features, seq_len]
        x = x.permute(0, 2, 1)
        
        if mask is not None:
            mask = mask.squeeze(-1) 

        # Encoder path
        encoder_outputs = []
        encoder_masks = []
        current_x = x
        current_mask = mask
        
        for i, encoder in enumerate(self.encoders):
            # Apply encoder
            enc_out = encoder(current_x, current_mask)
            encoder_outputs.append(enc_out)
            encoder_masks.append(current_mask)
            
            # Apply pooling (except for the last layer)
            if i < self.num_layers - 1:
                current_x = self.pool(enc_out)
                current_mask = self._downsample_mask(current_mask, factor=2)
        
        # Bottleneck
        bottleneck_mask = self._downsample_mask(encoder_masks[-1], factor=2)
        bottleneck_out = self.bottleneck(self.pool(encoder_outputs[-1]), bottleneck_mask)
        
        # Decoder path
        current_x = bottleneck_out
        
        for i in range(self.num_layers):
            # Upsampling
            current_x = self.upsample_layers[i](current_x)
            
            # Get corresponding encoder output and mask
            enc_idx = self.num_layers - 1 - i
            enc_out = encoder_outputs[enc_idx]
            enc_mask = encoder_masks[enc_idx]
            
            # Apply mask to upsampled features
            current_x = self._apply_mask_to_conv(current_x, enc_mask)
            
            # Apply attention if this layer has attention
            layer_num = enc_idx + 1  # Convert to 1-indexed
            if layer_num in self.attention_layers:
                att_gate = self.attention_gates[f'layer_{layer_num}']
                attended_features = att_gate(g=current_x, x=enc_out, mask=enc_mask)
                current_x = current_x + attended_features
                
                # Store attention weights
                self.attention_weights[f'level{layer_num}'] = att_gate.last_attention_weights
            else:
                # Simple skip connection without attention
                current_x = current_x + enc_out
            
            # Apply decoder block
            current_x = self.decoders[i](current_x, enc_mask)
        
        # Final output
        output = self.final_conv(current_x)
        output = self._apply_mask_to_conv(output, mask)
        presigmoid_output = output.permute(0, 2, 1).squeeze(-1)
        output = torch.sigmoid(presigmoid_output)
        if mask is not None:
            output = output * mask.float()

        classifical_output = self.categorical_conv(current_x)
        classifical_output = self._apply_mask_to_conv(classifical_output, mask)
        binned_output      = presigmoid_output - self.logit_threshold
        classifical_output = self.postprocess(classifical_output, mask)

        return {"regression"      : output,
                "classification"  : classifical_output,
                "binarization"    : binned_output,
        }
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return attention weights for visualization."""
        return {k: v.cpu().numpy() for k, v in self.attention_weights.items()}
    

class HFModelWithUNet(nn.Module):
    """HF backbone + Attention U-Net head."""

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
            print(f"[models] Backbone + LoRA (rank={_lora_rank})")
            self.backbone = backbone  # skip torch.compile for peft compatibility
        else:
            if getattr(config, "freeze_backbone", False):
                for _p in backbone.parameters():
                    _p.requires_grad_(False)
                _n = sum(_p.numel() for _p in backbone.parameters())
                print(f"[models] Backbone FROZEN ({_n/1e6:.1f}M params)")
            print("COMPILING MODEL...")
            self.backbone = torch.compile(backbone)

        embed_dim = self.ntv3_config.embed_dim

        # UNet config: input_features = embed_dim + 1 (condition)
        unet_config = SimpleNamespace(
            input_features   = embed_dim + 1,
            num_features     = getattr(config, "unet_num_features", 64),
            dropout_prob     = getattr(config, "unet_dropout_prob", 0.1),
            num_layers       = getattr(config, "unet_num_layers", 3),
            attention_layers = getattr(config, "unet_attention_layers", None),
            init_threshold   = config.init_threshold,
        )
        self.unet = AttentionUNet1D(unet_config)
        self.model_name = model_name

    def forward(self, tokens: torch.Tensor, conditions: torch.Tensor,
                mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]               # [B, L, embed_dim]

        x = torch.cat([embedding, conditions], dim=-1)      # [B, L, embed_dim+1]

        return self.unet(x, mask=mask)      

class Transformer(nn.Module):
    """Vanilla Transformer for sequence-to-sequence regression (4D → 1D), assuming input already has position info."""

    def __init__(self, config):
        super().__init__()
        input_features   = config.input_features          
        d_model          = config.d_model             
        nhead            = config.nhead               
        num_layers       = config.num_layers          
        num_features     = config.num_features            
        dropout_prob     = config.dropout_prob        
        batch_first      = config.batch_first          
        attention_layers = config.attention_layers    

        self.d_model = d_model

        # input projection (4 → d_model)
        self.input_proj = nn.Linear(input_features, d_model)

        # transformer encoder (no extra positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=num_features,
            dropout=dropout_prob,
            batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        # output projection (d_model → 1)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: (B, L, input_dim) — already contains positional features
            src_key_padding_mask: (B, L), True where padding
        Returns:
            (B, L) predictions
        """
        x = self.input_proj(src)                                   # (B, L, d_model)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)
        x = self.norm(x)
        x = self.output_proj(x)                                    # (B, L, 1)
        return x.squeeze(-1)                                       # (B, L)

class PositionalEmbedding(torch.nn.Module):
  def __init__(self, d_embed: int, max_seq_len: int):
    assert d_embed % 2 == 1, "only odd dimensional positional embeddings are supported"
    assert d_embed > 1, "positional embedding must be at least 3"
    super().__init__()

    t = torch.linspace(start=0, end=1, steps=max_seq_len)[:, None]

    tp = torch.linspace(start=0, end=max_seq_len - 1, steps=max_seq_len)[:, None]
    K = (d_embed - 1) // 2
    k = torch.linspace(start=0, end=K - 1, steps=K)[None, :]
    z = torch.exp(2 * math.pi * 1j * k * tp / max_seq_len)
    self.pos_emb = torch.nn.Parameter(torch.cat([t, z.real, z.imag], dim=-1), requires_grad=True)

  def forward(self, L: int) -> torch.Tensor:
    return self.pos_emb[:L]


class Sin(torch.nn.Module):
  def __init__(self, d_model: int, omega: int = 8, trainable: bool = False):
    super().__init__()
    self.freq = torch.nn.Parameter(omega * torch.ones(1, d_model), requires_grad=trainable)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sin(self.freq * x)

class Projection(torch.nn.Module):
  def __init__(self, d_model: int, N: int, conv_len: int):
    super().__init__()
    self.d_model = d_model
    self.N = N
    self.linear = torch.nn.Linear(d_model, d_model * (N + 1))
    self.conv = torch.nn.Conv1d(
      in_channels=d_model * (N + 1),
      out_channels=d_model * (N + 1),
      kernel_size=conv_len,
      groups=d_model * (N + 1),  # Depthwise convolution
      padding=conv_len - 1,
    )

  def forward(self, u: torch.Tensor):
    z = self.linear(u)
    z = z.transpose(1, 2)  # Channels (embedding dim) needs to come first

    L = z.shape[2]
    z = self.conv(z)[..., :L]

    x = torch.split(z, self.d_model, dim=1)
    return x

class FFTConv(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(
      self,
      h: torch.Tensor,
      x: torch.Tensor,
      B: torch.Tensor
    ) -> torch.Tensor:
    #orig_dtype = x.dtype
    #h = h.float()
    #x = x.float()
    #B = B.float()
    L = h.shape[-1]
    h_f = torch.fft.rfft(h, n=2 * L, norm="forward")
    x_f = torch.fft.rfft(x.to(dtype=h.dtype), n=2 * L)
    y = torch.fft.irfft(h_f * x_f, n=2 * L, norm="forward")[..., :L]
    y = y + x * B
    y = y.to(dtype=h.dtype)  # y is ComplexFloat but we need it to be float
    return y#.to(dtype=orig_dtype)

class HyenaBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_input = Projection(config.d_model, config.order, config.short_conv_size)
        self.proj_output = torch.nn.Linear(config.d_model, config.d_model)
        self.filter = HyenaFilter(
            d_model = config.d_model,
            d_mlp = config.d_filter_mlp,
            d_embed = config.d_embed,
            N = config.order,
            n_layers = config.n_filter_layers,
            max_seq_len = config.context_length,
            omega = config.omega,
            use_mlp=config.use_mlp
        )
        #self.dropout = torch.nn.Dropout(config.pdrop_hyena)
        self.fft_conv = FFTConv()
        self.B = torch.nn.Parameter(torch.randn((config.order, 1, config.d_model, 1)))

    #def _ckpt_proj(self, u: torch.Tensor):
    #   return self.proj_input(u)

    #def _ckpt_step(self, v: torch.Tensor, x_i: torch.Tensor, h_i: torch.Tensor, B_i: torch.Tensor, valid_1: torch.Tensor):
    #    return (v + x_i * self.fft_conv(h_i, v, B_i)) * valid_1

    def forward(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.to(dtype=torch.bool)              # (B, L), True = valid token
        valid_1 = valid.unsqueeze(1)                   # (B, 1, L)

        u = u * valid.unsqueeze(-1).to(u.dtype)        # (B, L, d_model)
        L = u.shape[1]

        *x, v = self.proj_input(u) #checkpoint.checkpoint(self._ckpt_proj, u, use_reentrant=False) # each: (B, d_model, L)
        v = (v + u.transpose(1, 2)) * valid_1          # (B, d_model, L)

        h = self.filter(L)                             # (N, d_model, L)

        for i, x_i in enumerate(x):
          x_i = x_i * valid_1
          x_i = torch.nn.functional.normalize(x_i, dim=1, eps=1e-5) * valid_1
          h_i = h[i].unsqueeze(0)                      # (1, d_model, L)
          v = (v + x_i * self.fft_conv(h_i, v, self.B[i])) * valid_1 #checkpoint.checkpoint(self._ckpt_step, v, x_i, h_i, self.B[i], valid_1, use_reentrant=False)

        v = v.transpose(1, 2)                          # (B, L, d_model)
        y = v + self.proj_output(v)
        y = y * valid.unsqueeze(-1).to(y.dtype)
        return y

class Window(torch.nn.Module):
  def __init__(
      self,
      d_model: int,
      max_seq_len: int,
      fast_decay_pct: float = 0.3,
      slow_decay_pct: float = 1.5,
      target: float = 1e-2,
    ):
    super().__init__()
    self.b = torch.nn.Parameter(torch.zeros((1, d_model, 1)))
    min_decay = math.log(target) / slow_decay_pct
    max_decay = math.log(target) / fast_decay_pct
    self.alphas = torch.nn.Parameter(
      torch.linspace(
        start=min_decay,
        end=max_decay,
        steps=d_model
      )[None, :, None]
    )
    self.register_buffer(
      "t",
      torch.linspace(start=0, end=1, steps=max_seq_len)[None, None, :],
      persistent=False,
    )
  def forward(self, x):
    L = x.shape[2]
    c = torch.exp(self.alphas * self.t)[:, :, :L]
    x = x * (c + self.b)
    return x

class HyenaFilter(torch.nn.Module):
  def __init__(
      self,
      d_model: int,
      d_mlp: int,
      d_embed: int,
      N: int,
      n_layers: int = 4,
      max_seq_len: int = 128,
      omega: int = 8,
      use_mlp:bool = False,
    ):
    assert n_layers >= 2, "n_layers must be at least 2"
    super().__init__()

    self.N = N
    self.d_model = d_model
    self.use_mlp = use_mlp
    self.window = Window(d_model, max_seq_len)
    if self.use_mlp:
        self.pos_emb = PositionalEmbedding(d_embed, max_seq_len)
    
        layers = [torch.nn.Linear(d_embed, d_mlp), Sin(d_mlp, omega)]
        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(d_mlp, d_mlp), Sin(d_mlp, omega)]
        layers += [torch.nn.Linear(d_mlp, N * d_model, bias=False)]
        self.mlp = torch.nn.Sequential(*layers)
    else:
       self.h = torch.nn.Parameter(torch.randn((N, d_model, max_seq_len)))

  def forward(self, L: int) -> torch.Tensor:
    if self.use_mlp:
        z = self.pos_emb(L)                         # z: (1, L, d_embed) # we build t in window
        h = self.mlp(z)                             # (1, L, N*d_model)
        h = h.transpose(0, 1).reshape(self.N, self.d_model, L)
    else:
        h = self.h[:, :, :L]
    h = self.window(h)
    h = h / torch.norm(h, dim=-2, p=1, keepdim=True).clamp_min(1e-12)
    return h

class Hyena(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.tok_emb = torch.nn.Linear(config.input_features, config.d_model)
    self.pos_emb = torch.nn.Parameter(
      torch.randn(1, config.context_length, 1)
    )
    self.drop = torch.nn.Dropout(config.dropout_prob)
    self.layers = torch.nn.Sequential(*[
      HyenaBlock(config) for _ in range(config.num_layers)
    ])
    self.ln = torch.nn.LayerNorm(config.d_model)
    self.head = torch.nn.Linear(
       in_features  = config.d_model,
       out_features = 1             ,
       bias         = False
    )
    # input embedding and logit output weights are tied
    #self.head.weight = self.tok_emb.weight

  def forward(self, x, mask) -> torch.Tensor:
    token_embeddings = self.tok_emb(x)
    position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]\
    .expand(-1, -1, self.config.d_model)
    x = self.drop(token_embeddings + position_embeddings)
    for layer in self.layers:
        x = layer(x, mask)
    x = self.ln(x)
    x = self.head(x)
    return x.squeeze(-1) 
  

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

BASE_MODEL_REGISTRY = {
    'AttentionUNet1D': AttentionUNet1D,
    'Transformer': Transformer,
    'Hyena': Hyena,
    'NTv3': HFModelWithHead,
    'NTv3mlp': HFModelWithMLP,
    'NTv3cnn': HFModelWithCNN,
    'NTv3unet': HFModelWithUNet,
}


def _available_model_names():
    return list(BASE_MODEL_REGISTRY.keys()) + [
        'DilatedCNN',
        'HyenaDNA',
        '<models_hybrid.*>',
        '<models_hybrid2.*>',
        '<ntv3_dilated_heads.*>',
    ]

def get_model(model_name: str, config) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model parameters
        
    Returns:
        Initialized model
    """
    if model_name in BASE_MODEL_REGISTRY:
        return BASE_MODEL_REGISTRY[model_name](config)

    if model_name == 'DilatedCNN':
        from models_dilated_cnn import DilatedCNN
        return DilatedCNN(config)

    if model_name == 'HyenaDNA':
        from hyenadna import HyenaOperator
        return HyenaOperator(config)

    from models_hybrid import HYBRID_MODELS
    if model_name in HYBRID_MODELS:
        return HYBRID_MODELS[model_name](config)

    from models_hybrid2 import HYBRID_MODELS_V2
    if model_name in HYBRID_MODELS_V2:
        return HYBRID_MODELS_V2[model_name](config)

    from ntv3_dilated_heads import NTV3_DILATED_MODELS
    if model_name in NTV3_DILATED_MODELS:
        return NTV3_DILATED_MODELS[model_name](config)

    raise ValueError(f"Model {model_name} not found in registry. Available: {_available_model_names()}")
