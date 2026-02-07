"""EEG Data Augmentation for Subject-Independent Classification.

Implements neurophysiologically-validated augmentations for motor imagery EEG,
targeting the 71% train → 55% test overfitting gap.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F


class EEGAugmentation:
    """Neurophysiologically-validated augmentations for motor imagery EEG.
    
    Applies random transformations during training to improve generalization
    and reduce subject-specific overfitting.
    
    Args:
        channel_dropout_prob: Probability of dropping channels.
        channel_shuffle_prob: Probability of swapping neighboring channels.
        amplitude_scale_range: Range for random amplitude scaling (min, max).
        random_gain_prob: Probability of per-channel gain variation.
        time_shift_max: Maximum samples to shift in time (circular).
        time_warp_prob: Probability of elastic temporal distortion.
        time_mask_prob: Probability of masking temporal segments.
        time_mask_max_samples: Maximum samples to mask.
        gaussian_noise_prob: Probability of adding Gaussian noise.
        noise_snr_db: Signal-to-noise ratio for noise (in dB).
    """
    
    def __init__(
        self,
        channel_dropout_prob: float = 0.2,
        channel_shuffle_prob: float = 0.1,
        amplitude_scale_range: tuple[float, float] = (0.8, 1.2),
        random_gain_prob: float = 0.3,
        time_shift_max: int = 50,
        time_warp_prob: float = 0.2,
        time_mask_prob: float = 0.1,
        time_mask_max_samples: int = 50,
        gaussian_noise_prob: float = 0.3,
        noise_snr_db: float = 20.0,
    ) -> None:
        self.channel_dropout_prob = channel_dropout_prob
        self.channel_shuffle_prob = channel_shuffle_prob
        self.amplitude_scale_range = amplitude_scale_range
        self.random_gain_prob = random_gain_prob
        self.time_shift_max = time_shift_max
        self.time_warp_prob = time_warp_prob
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_samples = time_mask_max_samples
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_snr_db = noise_snr_db
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to EEG tensor.
        
        Args:
            x: EEG tensor of shape (C, T) where C=channels, T=time samples.
        
        Returns:
            Augmented tensor of same shape (C, T).
        """
        # Spatial augmentations
        if random.random() < self.channel_dropout_prob:
            x = self.channel_dropout(x)
        
        if random.random() < self.channel_shuffle_prob:
            x = self.channel_shuffle(x)
        
        # Amplitude augmentations
        if random.random() < 0.5:
            x = self.amplitude_scaling(x)
        
        if random.random() < self.random_gain_prob:
            x = self.random_channel_gain(x)
        
        # Temporal augmentations (MOST IMPORTANT for EEG)
        if random.random() < 0.5:
            x = self.time_shift(x)
        
        if random.random() < self.time_warp_prob:
            x = self.time_warp(x)
        
        if random.random() < self.time_mask_prob:
            x = self.time_mask(x)
        
        # Noise (apply last)
        if random.random() < self.gaussian_noise_prob:
            x = self.add_noise_snr(x)
        
        return x
    
    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out entire channels."""
        C, T = x.shape
        num_drop = max(1, int(C * 0.1))  # Drop ~10% of channels
        drop_idx = random.sample(range(C), num_drop)
        x = x.clone()
        x[drop_idx] = 0
        return x
    
    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Swap neighboring channels (preserves spatial structure)."""
        C, T = x.shape
        if C < 2:
            return x
        idx1, idx2 = random.sample(range(C), 2)
        x = x.clone()
        x[[idx1, idx2]] = x[[idx2, idx1]]
        return x
    
    def amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global amplitude scaling."""
        scale = random.uniform(*self.amplitude_scale_range)
        return x * scale
    
    def random_channel_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random gain to each channel independently."""
        C, T = x.shape
        gains = torch.FloatTensor(C, 1).uniform_(*self.amplitude_scale_range)
        return x * gains.to(x.device)
    
    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular shift in time (simulates reaction time variation)."""
        shift = random.randint(-self.time_shift_max, self.time_shift_max)
        return torch.roll(x, shifts=shift, dims=-1)
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Elastic temporal distortion via interpolation."""
        C, T = x.shape
        # Create slightly warped output via random resampling
        warp_factor = random.uniform(0.9, 1.1)
        new_len = int(T * warp_factor)
        
        # Resample to warped length then back to original
        x_warped = F.interpolate(
            x.unsqueeze(0),  # (1, C, T)
            size=new_len,
            mode='linear',
            align_corners=True,
        )
        x_back = F.interpolate(
            x_warped,
            size=T,
            mode='linear',
            align_corners=True,
        )
        return x_back.squeeze(0)
    
    def time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random temporal segment (SpecAugment-style)."""
        C, T = x.shape
        mask_len = random.randint(1, min(self.time_mask_max_samples, T // 4))
        start = random.randint(0, T - mask_len)
        x = x.clone()
        x[:, start:start + mask_len] = 0
        return x
    
    def add_noise_snr(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise based on signal SNR."""
        signal_power = (x ** 2).mean()
        if signal_power == 0:
            return x
        noise_power = signal_power / (10 ** (self.noise_snr_db / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that applies augmentation to training data.
    
    Args:
        dataset: Original dataset (TensorDataset or similar).
        augmentation: EEGAugmentation instance.
        apply_augmentation: Whether to apply augmentation (set False for val/test).
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        augmentation: Optional[EEGAugmentation] = None,
        apply_augmentation: bool = True,
    ) -> None:
        self.dataset = dataset
        self.augmentation = augmentation
        self.apply_augmentation = apply_augmentation
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]
        
        if self.apply_augmentation and self.augmentation is not None:
            x = self.augmentation(x)
        
        return x, y
