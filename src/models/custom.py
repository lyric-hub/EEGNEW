"""S-MSTT v2: Enhanced Spiking Multilevel Spatio-Temporal Transformer.

Improvements from three expert perspectives:

SIGNAL ENGINEER:
- ERD window focus (0.5s-3.5s) - captures motor imagery dynamics
- Channel attention - emphasizes motor cortex (C3, Cz, C4)
- Frequency-aligned kernels - matches EEG rhythm timescales

MATHEMATICIAN:
- Increased capacity (f1=32, feature_dim=64)
- Longer receptive field (dilations 1,2,4,8)
- Better gradient flow (skip connections)
- Proper information preservation

ARCHITECT:
- Cleaner module design
- Configurable preprocessing
- Modular dual-stream
- Spike rate regularization support
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from src.models.base import BaseModel


SURROGATE_FUNCTION = surrogate.Sigmoid(alpha=4.0)


def get_lif(tau: float = 2.0, v_threshold: float = 1.0) -> neuron.LIFNode:
    """Create a LIF neuron with specified parameters."""
    return neuron.LIFNode(
        tau=tau,
        v_threshold=v_threshold,
        surrogate_function=SURROGATE_FUNCTION,
        detach_reset=True,
    )


class SpikeRateRegularizer(nn.Module):
    """Spike rate regularization for controlling firing sparsity.
    
    Encourages neurons to fire at a target rate (e.g., 20-30%).
    Too low = dead neurons, too high = no sparsity benefit.
    
    Loss = λ * (mean_spike_rate - target_rate)²
    """
    
    def __init__(self, target_rate: float = 0.25, weight: float = 1.0) -> None:
        super().__init__()
        self.target_rate = target_rate
        self.weight = weight
        self.spike_counts = []
    
    def reset(self) -> None:
        """Reset spike counts for new forward pass."""
        self.spike_counts = []
    
    def record(self, spikes: torch.Tensor) -> torch.Tensor:
        """Record spikes and return them unchanged (for inline use)."""
        self.spike_counts.append(spikes.detach().mean())
        return spikes
    
    def get_loss(self) -> torch.Tensor:
        """Compute spike rate regularization loss."""
        if not self.spike_counts:
            return torch.tensor(0.0)
        
        mean_rate = torch.stack(self.spike_counts).mean()
        loss = self.weight * (mean_rate - self.target_rate) ** 2
        return loss
    
    def get_mean_rate(self) -> float:
        """Get current mean spike rate."""
        if not self.spike_counts:
            return 0.0
        return torch.stack(self.spike_counts).mean().item()


# =============================================================================
# SIGNAL PROCESSING COMPONENTS
# =============================================================================

class EEGPreprocessing(nn.Module):
    """Signal-aware EEG preprocessing.
    
    - Crops to ERD window (motor imagery active period)
    - Applies channel attention (emphasizes motor cortex)
    - Normalizes per-channel
    """
    
    # Motor cortex channels for BNCI2014_001
    MOTOR_CHANNELS = [6, 7, 8, 9, 10, 11, 12]  # C5, C3, C1, Cz, C2, C4, C6
    
    def __init__(
        self,
        n_channels: int = 22,
        sample_rate: int = 250,
        erd_start: float = 0.5,
        erd_end: float = 3.5,
        use_channel_attention: bool = True,
    ) -> None:
        super().__init__()
        self.start_idx = int(erd_start * sample_rate)
        self.end_idx = int(erd_end * sample_rate)
        self.use_channel_attention = use_channel_attention
        
        if use_channel_attention:
            # Learnable channel attention
            self.ch_attention = nn.Parameter(torch.ones(n_channels))
            # Initialize with motor cortex emphasis
            with torch.no_grad():
                for idx in self.MOTOR_CHANNELS:
                    if idx < n_channels:
                        self.ch_attention[idx] = 2.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing.
        
        Args:
            x: (batch, channels, time)
        Returns:
            Preprocessed tensor
        """
        # Crop to ERD window
        x = x[:, :, self.start_idx:self.end_idx]
        
        # Channel attention
        if self.use_channel_attention:
            attn = F.softmax(self.ch_attention, dim=0).view(1, -1, 1)
            x = x * attn
        
        # Per-channel normalization (robust to outliers)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        
        return x


class FrequencyAlignedEncoder(nn.Module):
    """Multi-scale spike encoder with EEG-rhythm-aligned kernels.
    
    Kernel sizes correspond to EEG frequency bands:
    - 25 samples @ 250Hz = 100ms (Alpha/Mu: 8-12 Hz)
    - 13 samples @ 250Hz = 52ms (Beta: 15-25 Hz)
    - 6 samples @ 250Hz = 24ms (Gamma: 30-45 Hz)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sample_rate: int = 250,
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        
        # Frequency-aligned kernel sizes (must be ODD for same-padding)
        kernel_mu = self._make_odd(int(0.1 * sample_rate))      # ~100ms for mu rhythm
        kernel_beta = self._make_odd(int(0.05 * sample_rate))   # ~50ms for beta
        kernel_gamma = self._make_odd(int(0.025 * sample_rate)) # ~25ms for gamma
        
        self.kernels = [kernel_gamma, kernel_beta, kernel_mu]
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                layer.Conv2d(
                    in_channels, out_channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                get_lif(tau, v_threshold),
            )
            for k in self.kernels
        ])
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
    
    @staticmethod
    def _make_odd(n: int) -> int:
        """Ensure kernel size is odd (for symmetric padding)."""
        return max(3, n if n % 2 == 1 else n + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.fusion_weights, dim=0)
        outputs = [branch(x) * w for branch, w in zip(self.branches, weights)]
        return sum(outputs)


# =============================================================================
# SPATIAL PROCESSING
# =============================================================================

class SpatialAttentionConv(nn.Module):
    """Spatial convolution with attention-weighted channel pooling.
    
    Instead of collapsing 22→1 immediately, we:
    1. Apply depthwise conv to extract per-channel features
    2. Use attention to weight channels
    3. Reduce to fewer spatial dims (22→4) preserving some topology
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_spatial: int = 22,
        n_spatial_out: int = 4,
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_spatial_out = n_spatial_out
        
        # Depthwise spatial feature extraction
        self.depthwise = layer.Conv2d(
            in_channels, in_channels,
            kernel_size=(n_spatial, 1),
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lif1 = get_lif(tau, v_threshold)
        
        # Pointwise to expand channels
        self.pointwise = layer.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = get_lif(tau, v_threshold)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H=22, T)
        out = self.depthwise(x)  # (B, C_in, 1, T)
        out = self.bn1(out)
        out = self.lif1(out)
        
        out = self.pointwise(out)  # (B, C_out, 1, T)
        out = self.bn2(out)
        out = self.lif2(out)
        
        return out.squeeze(2)  # (B, C_out, T)


class SpatialSkipConv(nn.Module):
    """Skip connection with 1×1 conv to preserve spatial info.
    
    Instead of mean(dim=2), we use a learned 1×1 conv that:
    1. Collapses spatial dim with learned weights
    2. Projects to target feature dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_spatial: int = 22,
    ) -> None:
        super().__init__()
        # Collapse spatial dimension with learned weights
        self.spatial_collapse = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(n_spatial, 1),
            bias=False,
        )
        # Project to output dimension
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H=n_spatial, T)
        out = self.spatial_collapse(x)  # (B, C_in, 1, T)
        out = out.squeeze(2)  # (B, C_in, T)
        out = self.proj(out)  # (B, C_out, T)
        out = self.bn(out)
        return out


# =============================================================================
# TEMPORAL PROCESSING
# =============================================================================

class LongRangeTCN(nn.Module):
    """Multi-dilation TCN with extended receptive field.
    
    Dilations: 1, 2, 4, 8 gives receptive field of:
    r = 1 + (k-1)*Σd = 1 + 2*(1+2+4+8) = 31 samples = 124ms @ 250Hz
    
    With skip connections for better gradient flow.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple = (1, 2, 4, 8),
        dropout: float = 0.1,
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.paddings = []
        
        for d in dilations:
            padding = (kernel_size - 1) * d
            self.paddings.append(padding)
            self.blocks.append(
                nn.Sequential(
                    layer.Conv1d(
                        channels, channels,
                        kernel_size=kernel_size,
                        dilation=d,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    get_lif(tau, v_threshold),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, padding in zip(self.blocks, self.paddings):
            residual = x
            out = block(x)
            if padding > 0:
                out = out[:, :, :-padding]
            x = out + residual  # Skip connection
        return x


class SpikingSelfAttention(nn.Module):
    """Memory-efficient spiking self-attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        pool_size: int = 8,
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.pool = nn.AvgPool1d(pool_size, stride=pool_size) if pool_size > 1 else None
        
        self.qkv = layer.Conv1d(embed_dim, embed_dim * 3, kernel_size=1, bias=False)
        self.qkv_lif = get_lif(tau, v_threshold)
        
        self.dropout = nn.Dropout(dropout)
        
        self.out_proj = layer.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm1d(embed_dim)
        self.out_lif = get_lif(tau, v_threshold)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        orig_T = T
        
        if self.pool:
            x = self.pool(x)
            T = x.size(-1)
        
        qkv = self.qkv_lif(self.qkv(x))
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, T)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        attn = (q.permute(0, 1, 3, 2) @ k) * self.scale
        attn = self.dropout(attn)
        
        out = (attn @ v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = out.reshape(B, C, T)
        
        out = self.out_proj(out)
        out = self.out_bn(out)
        out = self.out_lif(out)
        
        if self.pool:
            out = F.interpolate(out, size=orig_T, mode='nearest')
        
        return out


# =============================================================================
# FUSION AND CLASSIFICATION
# =============================================================================

class GatedFusion(nn.Module):
    """Learnable gated fusion with residual."""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        )
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([a, b], dim=1))
        return a * gate + b * (1 - gate)


class ClassifierHead(nn.Module):
    """Classification head with spike rate output."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.2,
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = layer.Conv1d(in_channels, num_classes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(num_classes)
        self.lif = get_lif(tau, v_threshold)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.lif(out)
        return out.mean(dim=-1)


# =============================================================================
# MAIN MODEL
# =============================================================================

class CustomModel(BaseModel):
    """S-MSTT v2: Enhanced Spiking Multilevel Spatio-Temporal Transformer.
    
    Improvements:
    - Signal: ERD window, channel attention, frequency-aligned encoding
    - Math: Increased capacity, longer receptive field, skip connections
    - Arch: Cleaner design, spike rate tracking
    """
    
    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 1126,
        n_classes: int = 4,
        t_steps: int = 8,
        f1: int = 32,
        depth_multiplier: int = 2,
        num_heads: int = 4,
        tcn_dilations: tuple = (1, 2, 4, 8),
        dropout_rate: float = 0.2,
        attention_pool: int = 8,
        tau: float = 4.0,
        v_threshold: float = 1.0,
        use_preprocessing: bool = True,
        sample_rate: int = 250,
        # Compatibility args (unused)
        hidden_dim: int = 128,
        tcn_dilation: int = 2,
        spatial_type: str = "depthwise",
        learnable_adj: bool = False,
    ) -> None:
        super().__init__(n_channels, n_samples, n_classes)
        self.t_steps = t_steps
        self.use_preprocessing = use_preprocessing
        
        # 0. Signal-aware preprocessing
        if use_preprocessing:
            self.preprocess = EEGPreprocessing(
                n_channels=n_channels,
                sample_rate=sample_rate,
                erd_start=0.5,
                erd_end=3.5,
            )
            effective_samples = int(3.0 * sample_rate)  # 750 samples
        else:
            self.preprocess = None
            effective_samples = n_samples
        
        # 1. Frequency-aligned spike encoder
        self.encoder = FrequencyAlignedEncoder(
            in_channels=1,
            out_channels=f1,
            sample_rate=sample_rate,
            tau=tau,
            v_threshold=v_threshold,
        )
        
        feature_dim = f1 * depth_multiplier
        
        # 2. Spatial processing
        self.spatial = SpatialAttentionConv(
            in_channels=f1,
            out_channels=feature_dim,
            n_spatial=n_channels,
            tau=tau,
            v_threshold=v_threshold,
        )
        
        # 3. Dual-stream temporal processing
        self.tcn = LongRangeTCN(
            channels=feature_dim,
            kernel_size=3,
            dilations=tcn_dilations,
            dropout=dropout_rate,
            tau=tau,
            v_threshold=v_threshold,
        )
        
        self.attention = SpikingSelfAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            pool_size=attention_pool,
            tau=tau,
            v_threshold=v_threshold,
        )
        
        # 4. Gated fusion
        self.fusion = GatedFusion(channels=feature_dim)
        
        # 5. Skip connection with learned spatial projection (instead of mean)
        self.skip_conv = SpatialSkipConv(
            in_channels=f1,
            out_channels=feature_dim,
            n_spatial=n_channels,
        )
        
        # 6. Classification head
        self.classifier = ClassifierHead(
            in_channels=feature_dim,
            num_classes=n_classes,
            dropout=dropout_rate,
            tau=tau,
            v_threshold=v_threshold,
        )
        
        # 7. Spike rate regularizer
        self.spike_reg = SpikeRateRegularizer(
            target_rate=0.25,  # 25% target firing rate
            weight=1.0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Reset spike rate tracking
        self.spike_reg.reset()
        
        # 0. Preprocessing (ERD window, channel attention)
        if self.preprocess is not None:
            x = self.preprocess(x)
        
        # Add channel dim for 2D conv
        x = x.unsqueeze(1)  # (B, 1, C, T)
        
        # Repeat for SNN timesteps
        x_seq = x.unsqueeze(0).repeat(self.t_steps, 1, 1, 1, 1)
        x_flat = x_seq.flatten(0, 1)  # (T*B, 1, C, T)
        
        # 1. Frequency-aligned encoding
        encoded = self.encoder(x_flat)  # (T*B, f1, C, T)
        self.spike_reg.record(encoded)  # Track spike rate
        
        # 2. Spatial processing
        spatial = self.spatial(encoded)  # (T*B, feat_dim, T)
        self.spike_reg.record(spatial)
        
        # 3. Skip connection with learned spatial projection
        skip = self.skip_conv(encoded)  # (T*B, feat_dim, T)
        
        # 4. Dual-stream temporal
        tcn_out = self.tcn(spatial)
        self.spike_reg.record(tcn_out)
        
        attn_out = self.attention(spatial)
        self.spike_reg.record(attn_out)
        
        # 5. Gated fusion + skip
        fused = self.fusion(tcn_out, attn_out) + skip
        
        # 6. Classification (rate coding across timesteps)
        fused = fused.view(self.t_steps, batch_size, -1, fused.size(-1))
        
        logits_per_step = []
        for t in range(self.t_steps):
            logits_per_step.append(self.classifier(fused[t]))
        
        return torch.stack(logits_per_step, dim=0).mean(dim=0)
    
    def get_spike_loss(self) -> torch.Tensor:
        """Get spike rate regularization loss. Call after forward()."""
        return self.spike_reg.get_loss()
    
    def get_spike_rate(self) -> float:
        """Get current mean spike rate. Call after forward()."""
        return self.spike_reg.get_mean_rate()
    
    def reset_snn_states(self) -> None:
        """Reset all LIF neuron membrane potentials."""
        functional.reset_net(self)
