"""
S-MSTT v3.3: Spiking Multilevel Spatio-Temporal Transformer
Subject-Independent EEG Classification

v3.3 Improvements:
- DropPath (Stochastic Depth) for regularization
- Beta-biased multi-scale fusion [γ=0.2, β=0.6, μ=0.2]
- EEG data augmentation (channel/temporal/amplitude)
- Adaptive GroupNorm

Core Features:
- SVD-based Euclidean Alignment for subject independence
- Multi-scale temporal encoding (Mu/Beta/Gamma rhythms)
- Spiking Self-Attention with gradient-preserving residual
- GroupNorm instead of BatchNorm (no subject-specific statistics)
- Complete spike rate tracking across all LIF neurons
- Learned temporal aggregation over SNN timesteps
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, neuron, surrogate

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# SPIKE RATE TRACKING
# =============================================================================


class SpikeRateTracker:
    """Tracks spike rates from all LIF neurons for regularization."""

    def __init__(self, target_rate: float = 0.25, weight: float = 1.0) -> None:
        self.target_rate = target_rate
        self.weight = weight
        self.spike_counts: list[torch.Tensor] = []
        self.layer_rates: dict[str, list[float]] = {}

    def reset(self) -> None:
        self.spike_counts = []
        self.layer_rates = {}

    def record(self, spikes: torch.Tensor, layer_name: str = "default") -> torch.Tensor:
        rate = spikes.detach().mean()
        self.spike_counts.append(rate)
        if layer_name not in self.layer_rates:
            self.layer_rates[layer_name] = []
        self.layer_rates[layer_name].append(rate.item())
        
        # Assert binary spikes (spikingjelly LIF outputs should be 0 or 1)
        assert spikes.max() <= 1.0, f"Non-binary spikes in {layer_name}: max={spikes.max()}"
        return spikes

    def get_loss(self) -> torch.Tensor:
        if not self.spike_counts:
            return torch.tensor(0.0)
        mean_rate = torch.stack(self.spike_counts).mean()
        return self.weight * (mean_rate - self.target_rate) ** 2

    def get_mean_rate(self) -> float:
        if not self.spike_counts:
            return 0.0
        return torch.stack(self.spike_counts).mean().item()

    def get_layer_rates(self) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self.layer_rates.items() if v}


# =============================================================================
# EUCLIDEAN ALIGNMENT
# =============================================================================


class EuclideanAlignment(nn.Module):
    """SVD-based Euclidean Alignment for subject independence."""

    def __init__(
        self,
        channels: int,
        momentum: float = 0.1,
        warmup_batches: int = 10,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.warmup_batches = warmup_batches
        self.register_buffer("running_R", torch.eye(channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)

        B, C, T = x.shape
        x_mean = x.mean(dim=2, keepdim=True)
        x_centered = x - x_mean

        batch_cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / max(T - 1, 1)
        R_batch = batch_cov.mean(dim=0)

        if self.training:
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_R = (
                    (1 - self.momentum) * self.running_R
                    + self.momentum * R_batch.detach()
                )
            if self.num_batches_tracked < self.warmup_batches:
                # Use identity transform during warmup to maintain gradient flow
                whitening_mat = torch.eye(C, device=x.device, dtype=x.dtype)
                x_aligned = torch.matmul(whitening_mat, x_centered)
                return x_aligned.unsqueeze(1)

        # Detach running_R: EA is a preprocessing transform, not learned.
        # Gradients should flow through the whitening operation but not
        # update the covariance estimate (which is computed from data statistics).
        R = self.running_R.detach()
        try:
            U, S, Vh = torch.linalg.svd(R)
            S = torch.clamp(S, min=1e-5)
            whitening_mat = U @ torch.diag(S.pow(-0.5)) @ Vh
        except RuntimeError as e:
            logger.warning(f"SVD failed, using identity matrix: {e}")
            whitening_mat = torch.eye(C, device=x.device)

        x_aligned = torch.matmul(whitening_mat, x_centered)
        return x_aligned.unsqueeze(1)


# =============================================================================
# CHANNEL ATTENTION
# =============================================================================


class ChannelAttention(nn.Module):
    """Learnable channel attention with optional motor cortex bias."""

    def __init__(
        self,
        n_channels: int,
        motor_indices: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.attention = nn.Parameter(torch.ones(n_channels))
        if motor_indices:
            with torch.no_grad():
                for idx in motor_indices:
                    if idx < n_channels:
                        self.attention[idx] = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = F.softmax(self.attention, dim=0).view(1, 1, -1, 1)
        return x * attn


# =============================================================================
# DROP PATH (STOCHASTIC DEPTH)
# =============================================================================


class DropPath(nn.Module):
    """Stochastic Depth: randomly drops entire blocks during training.
    
    Paper: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
    """
    
    def __init__(self, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any tensor shape
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# =============================================================================
# MULTI-SCALE ENCODER
# =============================================================================


class MultiScaleTemporalEncoder(nn.Module):
    """Mu/Beta/Gamma rhythm-aligned encoding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sample_rate: int = 250,
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        kernel_mu = self._make_odd(int(0.1 * sample_rate))
        kernel_beta = self._make_odd(int(0.05 * sample_rate))
        kernel_gamma = self._make_odd(int(0.025 * sample_rate))

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, k), padding=(0, k // 2), bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),  # Adaptive GroupNorm
                neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan()),
            )
            for k in [kernel_gamma, kernel_beta, kernel_mu]
        ])
        # Beta-biased initialization: neuroscience evidence suggests beta rhythm
        # is most important for motor imagery. Still learnable via softmax.
        self.fusion_weights = nn.Parameter(torch.tensor([0.2, 0.6, 0.2]))
        self.names = ["gamma", "beta", "mu"]

    @staticmethod
    def _make_odd(n: int) -> int:
        return max(3, n if n % 2 == 1 else n + 1)

    def forward(self, x: torch.Tensor, tracker: Optional[SpikeRateTracker] = None) -> torch.Tensor:
        weights = F.softmax(self.fusion_weights, dim=0)
        outputs = []
        for i, (branch, w) in enumerate(zip(self.branches, weights)):
            out = branch(x) * w
            if tracker:
                tracker.record(out, f"enc_{self.names[i]}")
            outputs.append(out)
        return sum(outputs)


# =============================================================================
# SPIKING SELF-ATTENTION
# =============================================================================


class SpikingSelfAttention(nn.Module):
    """Spiking attention with gradient-preserving residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        tau: float = 2.0,
        dropout: float = 0.1,
        attn_residual_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_residual_ratio = attn_residual_ratio

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.q_lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())
        self.k_lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())
        self.v_lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())
        self.attn_lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())

        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.proj_norm = nn.LayerNorm(dim)
        self.proj_lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tracker: Optional[SpikeRateTracker] = None) -> torch.Tensor:
        T, B, N, D = x.shape

        q = self.q_lif(self.q_proj(x))
        k = self.k_lif(self.k_proj(x))
        v = self.v_lif(self.v_proj(x))

        if tracker:
            tracker.record(q, "ssa_q")
            tracker.record(k, "ssa_k")
            tracker.record(v, "ssa_v")

        q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale

        # Normalize attention with softmax over key dimension (proper distribution)
        attn_weights = F.softmax(attn_score, dim=-1)

        if self.attn_residual_ratio > 0:
            # Gradient-preserving: mix softmax weights with LIF spikes
            attn_spike = self.attn_lif(attn_weights)
            r = self.attn_residual_ratio
            attn_out = (1 - r) * attn_spike + r * attn_weights
        else:
            attn_out = self.attn_lif(attn_weights)

        if tracker:
            tracker.record(attn_out, "ssa_attn")

        attn_out = self.dropout(attn_out)
        out = attn_out @ v
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, N, D)

        out = self.proj_out(out)
        out = self.proj_norm(out)
        out = self.proj_lif(out)

        if tracker:
            tracker.record(out, "ssa_proj")

        return out


# =============================================================================
# SPIKING MLP
# =============================================================================


class SpikingMLP(nn.Module):
    """Feed-forward with LayerNorm."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        dropout: float = 0.1,
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.lif1 = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())

        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)
        self.norm2 = nn.LayerNorm(in_features)
        self.lif2 = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tracker: Optional[SpikeRateTracker] = None) -> torch.Tensor:
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.lif1(out)
        if tracker:
            tracker.record(out, "mlp_lif1")
        out = self.drop(out)

        out = self.fc2(out)
        out = self.norm2(out)
        out = self.lif2(out)
        if tracker:
            tracker.record(out, "mlp_lif2")

        return out


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================


class SBlock(nn.Module):
    """Transformer block with spike tracking and stochastic depth."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        tau: float = 2.0,
        attn_residual_ratio: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads, tau, dropout, attn_residual_ratio)
        self.mlp = SpikingMLP(dim, int(dim * mlp_ratio), dropout, tau)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, tracker: Optional[SpikeRateTracker] = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(x, tracker))
        x = x + self.drop_path(self.mlp(x, tracker))
        return x


# =============================================================================
# TOKENIZER
# =============================================================================


class SpatioTemporalTokenizer(nn.Module):
    """Converts EEG to spike tokens."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        time_points: int = 1125,
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, embed_dim, (in_channels, 1), bias=False)
        self.gn1 = nn.GroupNorm(8, embed_dim)  # GroupNorm for subject independence
        self.lif1 = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())

        self.temporal_conv = nn.Conv2d(embed_dim, embed_dim, (1, 25), stride=(1, 5), padding=(0, 12), bias=False)
        self.gn2 = nn.GroupNorm(8, embed_dim)  # GroupNorm for subject independence
        self.lif2 = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())

        self.pool = nn.AvgPool2d((1, 4))
        self.seq_len = time_points // (5 * 4)

    def forward(self, x: torch.Tensor, tracker: Optional[SpikeRateTracker] = None) -> torch.Tensor:
        x = self.lif1(self.gn1(self.spatial_conv(x)))
        if tracker:
            tracker.record(x, "tok_spatial")

        x = self.lif2(self.gn2(self.temporal_conv(x)))
        if tracker:
            tracker.record(x, "tok_temporal")

        x = self.pool(x)
        return x.squeeze(2).permute(0, 2, 1)


# =============================================================================
# LEARNED TEMPORAL AGGREGATION
# =============================================================================


class LearnedTemporalAggregation(nn.Module):
    """Attention-based aggregation over SNN time steps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N, D = x.shape
        x_mean = x.mean(dim=2)
        scores = self.proj(x_mean).squeeze(-1)
        weights = F.softmax(scores, dim=0).unsqueeze(-1).unsqueeze(-1)
        return (x * weights).sum(dim=0)


# =============================================================================
# MAIN MODEL
# =============================================================================


class CustomModel(BaseModel):
    """
    S-MSTT v3.3: Spiking Multilevel Spatio-Temporal Transformer
    Subject-Independent EEG Classification

    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples per trial
        n_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        t_steps: SNN simulation time steps
        tau: LIF neuron time constant
        dropout_rate: Dropout rate
        use_alignment: Enable Euclidean Alignment
        use_channel_attention: Enable channel attention
        use_multiscale: Enable multi-scale encoding
        use_learned_aggregation: Enable learned temporal aggregation
        alignment_momentum: EA momentum value
        spike_target_rate: Target spike rate for regularization
        attn_residual_ratio: Soft attention mix ratio for gradient flow
        drop_path_rate: Stochastic depth drop rate (0.0 = no drop)
        motor_indices: Channel indices for motor cortex attention bias
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 1125,
        n_classes: int = 4,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        t_steps: int = 16,  # Increased for better temporal integration
        tau: float = 2.0,
        dropout_rate: float = 0.1,
        use_alignment: bool = True,
        use_channel_attention: bool = True,
        use_multiscale: bool = True,
        use_learned_aggregation: bool = True,
        alignment_momentum: float = 0.1,
        spike_target_rate: float = 0.25,
        attn_residual_ratio: float = 0.1,
        drop_path_rate: float = 0.1,  # Stochastic depth for regularization
        motor_indices: Optional[list[int]] = None,
        # Legacy compatibility args (unused in v3.1)
        f1: int = 32,
        depth_multiplier: int = 2,
        tcn_dilations: tuple = (1, 2, 4, 8),
        attention_pool: int = 8,
        v_threshold: float = 1.0,
        use_preprocessing: bool = True,
        sample_rate: int = 250,
        hidden_dim: int = 128,
        tcn_dilation: int = 2,
        spatial_type: str = "depthwise",
        learnable_adj: bool = False,
    ) -> None:
        super().__init__(n_channels, n_samples, n_classes)

        self.t_steps = t_steps
        self.input_channels = n_channels
        self.use_multiscale = use_multiscale
        self.use_learned_aggregation = use_learned_aggregation

        # Default motor indices for BNCI2014_001
        if motor_indices is None:
            motor_indices = [6, 7, 8, 9, 10, 11, 12]

        # 1. Alignment
        self.align = EuclideanAlignment(n_channels, alignment_momentum) if use_alignment else nn.Identity()

        # 2. Channel Attention
        self.ch_attn = ChannelAttention(n_channels, motor_indices) if use_channel_attention else nn.Identity()

        # 3. Encoding
        if use_multiscale:
            self.encoder = MultiScaleTemporalEncoder(1, embed_dim, tau=tau)
            self.spatial_collapse = nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), bias=False)
            self.spatial_gn = nn.GroupNorm(min(8, embed_dim), embed_dim)  # Adaptive GroupNorm
            self.spatial_lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan())
            
            # Temporal pooling with configurable factor
            self.temporal_pool = nn.AvgPool2d((1, attention_pool))
            self.seq_len = n_samples // attention_pool
            self.tokenizer = None
        else:
            self.encoder = None
            self.tokenizer = SpatioTemporalTokenizer(n_channels, embed_dim, n_samples, tau)
            self.seq_len = self.tokenizer.seq_len

        # 4. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 5. Transformer (with linearly increasing drop path for deeper blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SBlock(embed_dim, num_heads, mlp_ratio, dropout_rate, tau, attn_residual_ratio, dpr[i])
            for i in range(depth)
        ])

        # 6. Aggregation
        self.temporal_agg = LearnedTemporalAggregation(embed_dim) if use_learned_aggregation else None

        # 7. Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(dropout_rate)
        self.cls_head = nn.Linear(embed_dim, n_classes)

        # 8. Tracker
        self.spike_tracker = SpikeRateTracker(spike_target_rate, 1.0)

        logger.info(f"S-MSTT v3.2 | seq_len={self.seq_len} | params={self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dim() == 3:
            B, C, T = x.shape
            assert C == self.input_channels, f"Expected {self.input_channels} channels, got {C}"
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            B, _, C, T = x.shape
            assert C == self.input_channels, f"Expected {self.input_channels} channels, got {C}"
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        functional.reset_net(self)
        self.spike_tracker.reset()

        # 1. Alignment
        x = self.align(x)

        # 2. Channel Attention
        x = self.ch_attn(x)

        # 3. Tokenization
        if self.use_multiscale:
            x = self.encoder(x, self.spike_tracker)
            x = self.spatial_lif(self.spatial_gn(self.spatial_collapse(x)))
            self.spike_tracker.record(x, "spatial_collapse")
            x = self.temporal_pool(x)
            tokens = x.squeeze(2).permute(0, 2, 1)
        else:
            tokens = self.tokenizer(x, self.spike_tracker)

        # Handle sequence length
        B, N, D = tokens.shape
        if N != self.seq_len:
            tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.seq_len).transpose(1, 2)

        tokens = tokens + self.pos_embed

        # 4. SNN Simulation
        x_seq = tokens.unsqueeze(0).repeat(self.t_steps, 1, 1, 1)
        for block in self.blocks:
            x_seq = block(x_seq, self.spike_tracker)

        # 5. Aggregation
        x_out = self.temporal_agg(x_seq) if self.use_learned_aggregation else x_seq.mean(0)
        x_gap = x_out.mean(1)

        # 6. Classification
        return self.cls_head(self.head_drop(self.norm(x_gap)))

    def get_spike_loss(self) -> torch.Tensor:
        return self.spike_tracker.get_loss()

    def get_spike_rate(self) -> float:
        return self.spike_tracker.get_mean_rate()

    def get_layer_rates(self) -> dict[str, float]:
        return self.spike_tracker.get_layer_rates()

    def reset_snn_states(self) -> None:
        functional.reset_net(self)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
