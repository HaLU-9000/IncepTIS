"""Self-contained exp038 NTv3 + attention U-Net architecture."""

import os
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForMaskedLM


def hf_token():
    return os.environ.get("HF_TOKEN")


def load_ntv3_backbone(model_name, config):
    """Instantiate NTv3 architecture without loading pretrained backbone weights."""
    if os.environ.get("INCEPTIS_LOAD_NTV3_PRETRAINED") == "1":
        return AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, config=config, token=hf_token())
    return AutoModelForMaskedLM.from_config(config, trust_remote_code=True)


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int, dropout=0.1):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv1d(f_g, f_int, 1), nn.BatchNorm1d(f_int))
        self.W_x = nn.Sequential(nn.Conv1d(f_l, f_int, 1), nn.BatchNorm1d(f_int))
        self.psi = nn.Sequential(nn.Conv1d(f_int, 1, 1), nn.BatchNorm1d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights = None

    def forward(self, g, x, mask=None):
        psi = self.dropout(self.relu(self.W_g(g) + self.W_x(x)))
        if mask is not None:
            psi = psi * mask.unsqueeze(1)
        psi = self.psi(psi)
        if mask is not None:
            psi = psi * mask.unsqueeze(1)
        self.last_attention_weights = psi.detach()
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.conv(x)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        x = self.dropout(self.relu(self.bn(x)))
        return x if mask is None else x * mask.unsqueeze(1)


class AttentionUNet1D(nn.Module):
    def __init__(self, input_features, num_features=64, dropout=0.1, num_layers=3,
                 attention_layers=(1, 2, 3), init_threshold=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.attention_layers = list(attention_layers)
        self.attention_weights = OrderedDict()
        self.encoders = nn.ModuleList()
        incoming = input_features
        for i in range(num_layers):
            outgoing = num_features * 2**i
            self.encoders.append(ConvBlock(incoming, outgoing, 7 if i == 0 else 3,
                                           3 if i == 0 else 1, dropout))
            incoming = outgoing
        self.pool = nn.MaxPool1d(2)
        bottleneck = num_features * 2**num_layers
        self.bottleneck = ConvBlock(num_features * 2**(num_layers - 1), bottleneck, dropout=dropout)
        self.attention_gates = nn.ModuleDict()
        for layer in self.attention_layers:
            width = num_features * 2**(layer - 1)
            self.attention_gates[f"layer_{layer}"] = AttentionGate(
                width, width, max(width // 2, num_features // 2), dropout)
        self.upsample_layers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            in_ch = bottleneck if i == 0 else num_features * 2**(num_layers - i)
            out_ch = num_features * 2**(num_layers - i - 1)
            self.upsample_layers.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                                       nn.Conv1d(in_ch, out_ch, 3, padding=1)))
            self.decoders.append(ConvBlock(out_ch, out_ch, dropout=dropout))
        self.final_conv = nn.Conv1d(num_features, 1, 1)
        self.categorical_conv = nn.Conv1d(num_features, 65, 1)
        self.logit_threshold = nn.Parameter(torch.logit(torch.tensor(init_threshold)))

    @staticmethod
    def down(mask):
        return None if mask is None else F.max_pool1d(mask.float().unsqueeze(1), 2).squeeze(1) > 0

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        if mask is not None:
            mask = mask.squeeze(-1)
        outputs, masks, current, current_mask = [], [], x, mask
        for i, encoder in enumerate(self.encoders):
            current = encoder(current, current_mask)
            outputs.append(current)
            masks.append(current_mask)
            if i < self.num_layers - 1:
                current, current_mask = self.pool(current), self.down(current_mask)
        current = self.bottleneck(self.pool(outputs[-1]), self.down(masks[-1]))
        for i in range(self.num_layers):
            current = self.upsample_layers[i](current)
            idx = self.num_layers - i - 1
            if masks[idx] is not None:
                current = current * masks[idx].unsqueeze(1)
            layer = idx + 1
            skip = (self.attention_gates[f"layer_{layer}"](current, outputs[idx], masks[idx])
                    if layer in self.attention_layers else outputs[idx])
            current = self.decoders[i](current + skip, masks[idx])
        logits = self.final_conv(current).permute(0, 2, 1).squeeze(-1)
        regression = torch.sigmoid(logits)
        classification = self.categorical_conv(current).permute(0, 2, 1)
        if mask is not None:
            regression = regression * mask.float()
            logits = logits * mask.float()
            classification = classification * mask.unsqueeze(-1).float()
        return {"regression": regression, "classification": classification,
                "binarization": logits - self.logit_threshold}


class Exp038Model(nn.Module):
    def __init__(self, model_name="InstaDeepAI/NTv3_100M_pre"):
        super().__init__()
        token = hf_token()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)
        self.backbone = load_ntv3_backbone(model_name, config)
        self.unet = AttentionUNet1D(config.embed_dim + 1)

    def forward(self, tokens, conditions, mask=None):
        result = self.backbone(input_ids=tokens, output_hidden_states=True)
        x = torch.cat([result.hidden_states[-1], conditions], dim=-1)
        return self.unet(x, mask=mask)
