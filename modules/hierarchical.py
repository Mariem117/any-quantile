import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalQuantilePredictor(nn.Module):
    """Wrap a backbone with a hierarchical quantile head.

    The wrapper turns generic backbone features into monotonic quantile outputs
    using either a Gaussian location/scale parameterization or an incremental
    construction that enforces ordered quantiles.
    """

    def __init__(self, backbone, hidden_dim=None, horizon_length=1, method="gaussian"):
        super().__init__()
        self.backbone = backbone
        self.method = method
        self.horizon = horizon_length

        # Allow hidden_dim to be None (lazy-init from features on first forward)
        def make_linear(out_dim):
            return nn.LazyLinear(out_dim) if hidden_dim is None else nn.Linear(hidden_dim, out_dim)

        if method == "gaussian":
            # Parametric approach (assumes Normal distribution)
            self.mu_head = make_linear(horizon_length)
            self.sigma_head = make_linear(horizon_length)

        elif method == "incremental":
            # Non-parametric increment-based approach
            self.median_head = make_linear(horizon_length)
            self.increment_head = make_linear(horizon_length * 10)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _extract_features(self, x, quantiles):
        # Prefer explicit feature method if available
        if hasattr(self.backbone, "get_features"):
            return self.backbone.get_features(x)
        if hasattr(self.backbone, "encode"):
            return self.backbone.encode(x)

        # Fall back to backbone forward; try with quantiles then without
        try:
            feats = self.backbone(x, quantiles)
        except TypeError:
            feats = self.backbone(x)

        # Flatten everything except batch
        if feats.dim() > 2:
            feats = feats.reshape(feats.shape[0], -1)
        return feats
    
    def forward(self, x, quantiles):
        """
        Args:
            x: Input features [B, T, ...]
            quantiles: Quantile levels [B, Q] or [Q]
            
        Returns:
            predictions: [B, H, Q] where H is horizon
        """
        # Get backbone features (robust to missing get_features)
        features = self._extract_features(x, quantiles)
        
        if self.method == "gaussian":
            return self._forward_gaussian(features, quantiles)
        elif self.method == "incremental":
            return self._forward_incremental(features, quantiles)
    
    def _forward_gaussian(self, features, quantiles):
        """Gaussian hierarchical quantile prediction"""
        B = features.shape[0]
        
        # Predict distribution parameters
        mu = self.mu_head(features)  # [B, H]
        log_sigma = self.sigma_head(features)  # [B, H]
        sigma = F.softplus(log_sigma) + 1e-6  # Ensure positive
        
        # Handle quantile shape
        if quantiles.dim() == 1:
            quantiles = quantiles[None, :].expand(B, -1)  # [B, Q]
        
        # Compute z-scores for quantiles (inverse CDF of standard normal)
        z = torch.erfinv(2 * quantiles - 1) * (2 ** 0.5)  # [B, Q]
        
        # Generate predictions: q(τ) = μ + σ * z(τ)
        predictions = mu[:, :, None] + sigma[:, :, None] * z[:, None, :]
        
        return predictions  # [B, H, Q]
    
    def _forward_incremental(self, features, quantiles):
        """Increment-based hierarchical quantile prediction"""
        B, H = features.shape[0], self.horizon
        Q = quantiles.shape[-1]
        
        # Predict median
        median = self.median_head(features)  # [B, H]
        
        # Predict increments (always positive)
        increments = F.softplus(self.increment_head(features))  # [B, H*10]
        increments = increments.reshape(B, H, 10)
        
        # Sort quantiles to ensure monotonicity
        sorted_q, sort_idx = torch.sort(quantiles, dim=-1)
        
        # Initialize predictions
        predictions = torch.zeros(B, H, Q, device=features.device)
        
        # Find median index in sorted quantiles
        median_idx = (sorted_q - 0.5).abs().argmin(dim=-1)
        
        # Set median
        for b in range(B):
            predictions[b, :, sort_idx[b, median_idx[b]]] = median[b]
        
        # Build lower quantiles (decreasing from median)
        for b in range(B):
            for i in range(median_idx[b] - 1, -1, -1):
                increment_idx = median_idx[b] - i - 1
                predictions[b, :, sort_idx[b, i]] = (
                    predictions[b, :, sort_idx[b, i + 1]] - 
                    increments[b, :, increment_idx % 10]
                )
        
        # Build upper quantiles (increasing from median)
        for b in range(B):
            for i in range(median_idx[b] + 1, Q):
                increment_idx = i - median_idx[b] - 1
                predictions[b, :, sort_idx[b, i]] = (
                    predictions[b, :, sort_idx[b, i - 1]] + 
                    increments[b, :, increment_idx % 10]
                )
        
        return predictions  # [B, H, Q] - guaranteed monotonic!
