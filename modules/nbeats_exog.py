import torch
import torch.nn as nn
import torch.nn.functional as F


class ExogenousEncoder(nn.Module):
    """Encode exogenous features into embedding space"""
    
    def __init__(self,
                 num_continuous: int = 4,  # temp, humidity, etc.
                 num_calendar: int = 4,  # hour, dow, month, weekend
                 embed_dim: int = 64):
        super().__init__()
        
        # Continuous feature normalization + projection
        self.continuous_proj = nn.Sequential(
            nn.Linear(num_continuous, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Calendar feature embeddings
        self.hour_embed = nn.Embedding(24, embed_dim // 4)
        self.dow_embed = nn.Embedding(7, embed_dim // 4)
        self.month_embed = nn.Embedding(12, embed_dim // 4)
        self.weekend_embed = nn.Embedding(2, embed_dim // 4)
    
    def forward(self, continuous, calendar):
        """
        Args:
            continuous: [B, T, num_continuous] - weather features
            calendar: [B, T, 4] - (hour, dow, month, is_weekend) as indices
        
        Returns:
            [B, T, embed_dim] - combined embedding
        """
        # Project continuous features
        cont_embed = self.continuous_proj(continuous)
        
        # Get calendar embeddings
        cal_embed = torch.cat([
            self.hour_embed(calendar[..., 0].long()),
            self.dow_embed(calendar[..., 1].long()),
            self.month_embed(calendar[..., 2].long()),
            self.weekend_embed(calendar[..., 3].long())
        ], dim=-1)
        
        return cont_embed + cal_embed


class NbeatsBlockWithExog(nn.Module):
    """N-BEATS block with exogenous feature conditioning"""
    
    def __init__(self,
                 num_layers: int,
                 layer_width: int,
                 size_in: int,
                 size_out: int,
                 exog_dim: int = 64):
        super().__init__()
        
        self.layer_width = layer_width
        
        # Build FC layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(size_in, layer_width))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(layer_width, layer_width))
        
        # Forward and backward projections
        self.forward_projection = nn.Linear(layer_width, size_out)
        self.backward_projection = nn.Linear(layer_width, size_in)
        
        # Condition film for quantile conditioning
        self.condition_film = nn.Linear(layer_width, layer_width * 2)
        
        # Project exogenous embeddings to layer width
        self.exog_projection = nn.Linear(exog_dim, layer_width)
        
        # Gating mechanism to control exog influence
        self.exog_gate = nn.Sequential(
            nn.Linear(layer_width * 2, layer_width),
            nn.Sigmoid()
        )
    
    def forward(self, x, condition, exog_embed):
        """
        Args:
            x: [B, Q, T] - input time series (already replicated for Q quantiles)
            condition: [B, Q, layer_width] - quantile conditioning
            exog_embed: [B, T, exog_dim] or None - exogenous feature embeddings
        """
        h = x
        
        # Pool exogenous features over time dimension if available
        exog_pooled = None
        if exog_embed is not None:
            exog_pooled = self.exog_projection(exog_embed.mean(dim=1))  # [B, layer_width]
            exog_pooled = exog_pooled.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, Q, layer_width]
        
        for i, layer in enumerate(self.fc_layers):
            h = F.relu(layer(h))
            
            if i == 0:
                # FiLM conditioning from quantile
                condition_film = self.condition_film(condition)
                offset, delta = condition_film[..., :self.layer_width], condition_film[..., self.layer_width:]
                h = h * (1 + delta) + offset
                
                # Gated addition of exogenous features (only if available)
                if exog_pooled is not None:
                    gate = self.exog_gate(torch.cat([h, exog_pooled], dim=-1))
                    h = h + gate * exog_pooled
        
        return self.forward_projection(h), self.backward_projection(h)


class NBEATSEXOG(nn.Module):
    """N-BEATS model with exogenous feature support"""
    
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, 
                 size_in: int, size_out: int, exog_dim: int = 64,
                 num_continuous: int = 4, num_calendar: int = 4):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out
        self.share = share
        
        # Exogenous encoder
        self.exog_encoder = ExogenousEncoder(
            num_continuous=num_continuous,
            num_calendar=num_calendar, 
            embed_dim=exog_dim
        )
        
        # Build N-BEATS blocks with exogenous support
        self.blocks = [NbeatsBlockWithExog(
            num_layers=num_layers,
            layer_width=layer_width,
            size_in=size_in + 1,  # +1 for quantile
            size_out=size_out,
            exog_dim=exog_dim
        )]
        
        if self.share:
            for i in range(self.num_blocks-1):
                self.blocks.append(self.blocks[0])
        else:
            for i in range(self.num_blocks-1):
                self.blocks.append(NbeatsBlockWithExog(
                    num_layers=num_layers,
                    layer_width=layer_width,
                    size_in=size_in + 1,
                    size_out=size_out,
                    exog_dim=exog_dim
                ))
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x: torch.Tensor, q: torch.Tensor, 
                continuous: torch.Tensor = None, calendar: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T] - input time series
            q: [B, Q] - quantile specifications  
            continuous: [B, T, num_continuous] - weather features
            calendar: [B, T, 4] - (hour, dow, month, is_weekend) as indices
        
        Returns:
            [B, H, Q] - forward predictions for H horizons and Q quantiles
        """
        Q = q.shape[-1]
        
        # Prepare input with quantile concatenation
        backcast = torch.cat([
            torch.repeat_interleave(x[:, None], repeats=Q, dim=1, output_size=Q), 
            q[..., None]
        ], dim=-1)
        
        # Encode exogenous features if provided
        exog_embed = None
        if continuous is not None and calendar is not None:
            exog_embed = self.exog_encoder(continuous, calendar)
        
        # Prepare conditioning for quantile
        condition = torch.repeat_interleave(q[..., None], repeats=self.layer_width, dim=-1, output_size=self.layer_width)
        
        output = 0.0
        for block in self.blocks:
            f, b = block(backcast, condition, exog_embed)
            output = output + f
            backcast = backcast - b
        
        return output.transpose(-1, -2)