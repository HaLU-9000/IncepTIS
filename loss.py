from typing import Optional, Callable, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import create_mask_from_lengths

class IgnoreInvalidLoss(nn.Module):
    """
    Base class for loss functions that ignore negative and NaN target values.
    Children must implement `_compute_loss(predictions, targets, valid_mask)`.
    """

    def _build_valid_mask(
        self,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = (targets >= 0) & ~torch.isnan(targets)
        if mask is not None:
            mask = mask.bool()
            while mask.ndim < valid.ndim:
                mask = mask.unsqueeze(-1)
            valid = valid & mask
        return valid

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `_compute_loss`.")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        valid_mask = self._build_valid_mask(targets, mask)
        return self._compute_loss(predictions, targets, valid_mask)


class MSELoss(IgnoreInvalidLoss):
    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss(reduction="none")

    def _compute_loss(self, predictions, targets, valid_mask):
        return self._mse(predictions[valid_mask], targets[valid_mask]).mean()


class MAELoss(IgnoreInvalidLoss):
    def __init__(self):
        super().__init__()
        self._mae = nn.L1Loss(reduction="none")

    def _compute_loss(self, predictions, targets, valid_mask):
        return self._mae(predictions[valid_mask], targets[valid_mask]).mean()


class HuberLoss(IgnoreInvalidLoss):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self._huber = nn.HuberLoss(reduction="none", delta=delta)

    def _compute_loss(self, predictions, targets, valid_mask):
        return self._huber(predictions[valid_mask], targets[valid_mask]).mean()

class CrossEntropyLoss(IgnoreInvalidLoss):
    def __init__(self):
        super().__init__()
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def _compute_loss(self, predictions, targets, valid_mask):
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.any(dim=-1)  # [B, L]
        
        _, _, C = predictions.shape
        predictions_valid = predictions[valid_mask]  # [M, C]
        targets_valid = targets[valid_mask].float()  # [M, C]

        return self._ce(predictions_valid, targets_valid).mean()
    

class BCELoss(IgnoreInvalidLoss):
    def __init__(self, pos_weight=None):
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def _compute_loss(self, predictions, targets, valid_mask):
        targets = targets.squeeze(-1)  
        # valid_mask may be [B, L, 1] — reduce to match
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.any(dim=-1)  # [B, L, 1] → [B, L]
        return self._bce(predictions[valid_mask], targets[valid_mask]).mean()


class FocalLoss(IgnoreInvalidLoss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _compute_loss(self, predictions, targets, valid_mask):
        targets = targets.squeeze(-1)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.any(dim=-1)

        # apply mask first, then compute everything on valid elements only
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        p = torch.sigmoid(predictions)
        ce = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # gamma < 1 has an infinite derivative at p_t == 1 without this clamp.
        modulating_factor = (1 - p_t).clamp_min(torch.finfo(p.dtype).eps) ** self.gamma
        focal = alpha_t * modulating_factor * ce
        return focal.mean()

class MulticlassFocalLoss(nn.Module):
    """
    Multiclass focal loss per Lin et al. (2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = softmax probability of the correct class.
    """
    def __init__(
        self,
        num_classes: int = 65,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,  # (num_classes,) or None for uniform
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer("alpha", torch.ones(num_classes) / num_classes)

    def forward(self, predictions, targets, mask=None, **kwargs):
        N, L, C = predictions.shape
        targets_ce = targets.argmax(dim=-1)       # [N, L]
        valid_mask = targets.sum(dim=-1) > 0      # [N, L]

        if mask is not None:
            valid_mask = valid_mask & mask.bool()

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # mask everything down to [M] or [M, C] first
        predictions_valid = predictions[valid_mask]   # [M, C]
        targets_ce_valid  = targets_ce[valid_mask]    # [M]

        probs        = torch.softmax(predictions_valid, dim=-1)             # [M, C]
        targets_oh   = F.one_hot(targets_ce_valid, num_classes=C).float()  # [M, C]
        p_t          = (probs * targets_oh).sum(dim=-1)                    # [M]
        alpha_t      = self.alpha.to(targets_ce_valid.device)[targets_ce_valid]  # [M]
        focal_weight = (1.0 - p_t) ** self.gamma                           # [M]
        ce           = -torch.log(p_t.clamp(min=1e-8))                     # [M]
        loss         = alpha_t * focal_weight * ce                         # [M]
        return loss.mean()


class MultiLoss(nn.Module):
    def __init__(self, loss_specs: dict[str, dict[str, Any]]):
        super().__init__()
        self.loss_modules = nn.ModuleDict({
            name: spec["loss"] for name, spec in loss_specs.items()
        })
        self.loss_specs = loss_specs

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        total_loss = 0.0
        loss_dict = {}

        for name, spec in self.loss_specs.items():
            if spec.get("weight", 1.0) == 0:
                continue

            pred_key = spec["pred_key"]
            target_getter = spec["target_getter"]
            loss_fn = self.loss_modules[name]
            weight = spec.get("weight", 1.0)
            extra_kwargs_fn = spec.get("extra_kwargs_fn", None)

            predictions = outputs[pred_key]
            targets = target_getter(batch, outputs)

            extra_kwargs = {}
            if extra_kwargs_fn is not None:
                extra_kwargs = extra_kwargs_fn(batch, outputs)

            loss_value = loss_fn(
                predictions=predictions,
                targets=targets,
                mask=mask,
                **extra_kwargs,
            )

            weighted_loss = weight * loss_value
            total_loss = total_loss + weighted_loss

            loss_dict[name] = loss_value.detach()
            loss_dict[f"{name}_weighted"] = weighted_loss.detach()
        loss_dict["total"] = total_loss.detach()

        return total_loss, loss_dict
    
def compute_zero_output_loss(config, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[0] if "NTv3" in config.model else batch
            batch_lengths = batch[1].to(device) if "NTv3" in config.model else batch_data["lengths"].to(device)

            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }
            B, L = batch_on_device["tokens"].shape[:2]
            mask = create_mask_from_lengths(batch_lengths, L, device)

            zero_outputs = {
                "regression":     torch.zeros(B, L, device=device),
                "classification": torch.cat([
                    torch.full((B, L, 1),  10.0, device=device),
                    torch.full((B, L, 64), 0.0,  device=device),
                ], dim=-1),
                "binarization":   torch.zeros(B, L, device=device),
            }

            loss, loss_dict = criterion(outputs=zero_outputs, batch=batch_on_device, mask=mask)
            if not torch.isnan(loss) and loss.item() > 0:
                total_loss += loss.item()
                count += 1
    
    return total_loss / count if count > 0 else 0.0
