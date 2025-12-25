import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicityLoss(nn.Module):
    """Penalize crossings between adjacent quantile predictions.

    When predicting multiple quantiles, higher quantile levels should map to
    higher values. This loss softly penalizes violations with an optional
    positive margin.
    """

    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.margin = margin
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, quantiles: torch.Tensor):
        """Compute monotonicity penalty.

        Args:
            predictions: Tensor shaped [B, H, Q] with predicted values for Q quantiles.
            quantiles: Tensor shaped [B, Q] or [B, 1, Q] containing quantile levels.
        Returns:
            Scalar loss or tensor of violations depending on reduction.
        """
        if quantiles.dim() == 2:
            quantiles = quantiles.unsqueeze(1)
        elif quantiles.dim() != 3:
            raise ValueError("quantiles must be 2D or 3D with last dim = Q")

        if predictions.shape[-1] != quantiles.shape[-1]:
            raise ValueError("predictions and quantiles must share the last dimension")

        # Sort quantiles and align predictions to that order
        _, sort_idx = quantiles.sort(dim=-1)
        sort_idx_expanded = sort_idx.expand_as(predictions)
        pred_sorted = predictions.gather(-1, sort_idx_expanded)

        # Adjacent differences should be positive (optionally with a margin)
        diffs = pred_sorted[..., 1:] - pred_sorted[..., :-1]
        violations = F.relu(self.margin - diffs)

        if self.reduction == 'mean':
            return violations.mean()
        if self.reduction == 'sum':
            return violations.sum()
        return violations
