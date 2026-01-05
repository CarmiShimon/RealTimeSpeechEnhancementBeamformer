import torch
import torch.nn as nn
from omegaconf import DictConfig


class OnlineNorm(nn.Module):
    """
    Cleaned Online Normalization using Hydra config.
    Applies exponential moving average (EMA) for mean and variance tracking.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # 1. Config parameters
        # Features = (n_fft // 2 + 1)
        self.feat_size = cfg.audio.n_fft // 2 + 1
        fs = cfg.audio.sample_rate
        hop = cfg.audio.hop_length
        # Default Tpsd to 0.1 if not in config
        t_psd = cfg.get('norm', {}).get('Tpsd', 0.1)

        # 2. Calculate Smoothing Factor (alpha)
        # alpha = 1 - exp(-(Time_per_frame) / Time_constant)
        alpha = 1 - torch.exp(-torch.tensor(hop / fs) / t_psd)
        self.register_buffer('alpha', alpha)

        # 3. Permanent Buffers (Avoid re-registering in forward)
        # Shape: (1, 1, F) to allow broadcasting across Batch and Time
        self.register_buffer('running_mean', torch.zeros(1, 1, self.feat_size))
        self.register_buffer('running_var', torch.ones(1, 1, self.feat_size))

        self.eps = 1e-12

    def forward(self, x, mode='batch'):
        """
        Args:
            x: (B, T, F) Input spectrogram
            mode: 'batch' (for training/offline) or 'stream' (for real-time inference)
        """
        if mode == 'batch':
            return self._process_batch(x)
        else:
            return self._process_stream(x)

    def _process_batch(self, x):
        """Vectorized-style recursive update for training."""
        B, T, F = x.shape

        # Initialize outputs
        means = torch.zeros_like(x)
        vars = torch.ones_like(x)

        # We start with the current running stats
        curr_mean = self.running_mean.expand(B, -1, -1)  # (B, 1, F)
        curr_var = self.running_var.expand(B, -1, -1)

        # Recursive update (Causal)
        for t in range(T):
            frame = x[:, t:t + 1, :]  # (B, 1, F)

            # Update Mean: m = (1-a)m + a*x
            curr_mean = (1 - self.alpha) * curr_mean + self.alpha * frame
            # Update Var: v = (1-a)v + a*(x-m)^2
            curr_var = (1 - self.alpha) * curr_var + self.alpha * ((frame - curr_mean) ** 2)

            means[:, t:t + 1, :] = curr_mean
            vars[:, t:t + 1, :] = curr_var

        # Update global buffers with the last frame of the batch (no_grad for safety)
        with torch.no_grad():
            self.running_mean.copy_(means[:, -1:, :].mean(dim=0, keepdim=True))
            self.running_var.copy_(vars[:, -1:, :].mean(dim=0, keepdim=True))

        # Normalize (Using your specific formula)
        return (x - means) / (vars * 4 + self.eps)

    def _process_stream(self, x):
        """Single-step update for real-time streaming."""
        # x is (B, 1, F) or (B, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Update stats
        self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * x
        self.running_var = (1 - self.alpha) * self.running_var + self.alpha * ((x - self.running_mean) ** 2)

        return (x - self.running_mean) / (self.running_var * 4 + self.eps)

    def reset_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)

    def state_dict(self, *args, **kwargs):
        """Overridden to prevent saving running stats in checkpoints."""
        d = super().state_dict(*args, **kwargs)
        d.pop('running_mean', None)
        d.pop('running_var', None)
        return d