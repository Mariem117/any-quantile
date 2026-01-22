import numpy as np
import torch


class AdaptiveQuantileSampler:
    """Sample quantiles with probability proportional to recent loss."""

    def __init__(self, num_bins: int = 100, momentum: float = 0.99, temperature: float = 1.0, min_prob: float = 0.001):
        self.num_bins = num_bins
        self.momentum = momentum
        self.temperature = temperature
        self.min_prob = min_prob
        self.loss_estimates = np.ones(num_bins, dtype=np.float32)
        self.bin_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)

    def update(self, quantiles: torch.Tensor, losses: torch.Tensor) -> None:
        """Update running loss estimates per quantile bin."""
        quantiles_np = quantiles.detach().cpu().numpy().flatten()
        losses_np = losses.detach().cpu().numpy().flatten()
        bin_indices = np.digitize(quantiles_np, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        for bin_idx, loss in zip(bin_indices, losses_np):
            self.loss_estimates[bin_idx] = self.momentum * self.loss_estimates[bin_idx] + (1.0 - self.momentum) * loss

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample quantiles; higher-loss bins are more likely."""
        logits = self.loss_estimates / max(self.temperature, 1e-6)
        probs = np.exp(logits - logits.max())  # stable softmax
        probs = np.maximum(probs, self.min_prob)
        probs = probs / probs.sum()
        bins = np.random.choice(self.num_bins, size=batch_size, p=probs)
        q_low = self.bin_edges[bins]
        q_high = self.bin_edges[bins + 1]
        quantiles = q_low + np.random.rand(batch_size) * (q_high - q_low)
        return torch.tensor(quantiles, dtype=torch.float32)
