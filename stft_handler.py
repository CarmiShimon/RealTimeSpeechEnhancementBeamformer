import torch
import torch.nn as nn
from omegaconf import DictConfig


class STFTHandler(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.n_fft = cfg.audio.n_fft
        self.hop_length = cfg.audio.hop_length

        # Use a Hann window by default, moved to the correct device during forward
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def transform(self, x):
        """
        Transforms time-domain signal to STFT.
        Args:
            x: (B, C, L) where L is signal length
        Returns:
            complex_spec: (B, C, T, F)
        """
        batch_size, num_channels, length = x.shape

        # Reshape to treat channels as batch for torch.stft
        x = x.view(batch_size * num_channels, -1)

        # Output is (Batch*Channels, F, T)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
            center=True
        )

        # Reshape back to (B, C, F, T)
        _, num_freqs, num_frames = stft.shape
        stft = stft.view(batch_size, num_channels, num_freqs, num_frames)

        # Permute to your preferred (B, C, T, F) for the beamformer
        return stft.permute(0, 1, 3, 2)

    def inverse(self, x_f):
        """
        Transforms STFT back to time-domain.
        Args:
            x_f: (B, T, F) or (B, C, T, F) complex spectrogram
        Returns:
            x_t: (B, L) or (B, C, L) waveform
        """
        # If input is (B, T, F), add a dummy channel dim
        if x_f.dim() == 3:
            x_f = x_f.unsqueeze(1)

        batch_size, num_channels, num_frames, num_freqs = x_f.shape

        # Permute to (B*C, F, T) for torch.istft
        x_f = x_f.permute(0, 1, 3, 2).reshape(batch_size * num_channels, num_freqs, num_frames)

        waveform = torch.istft(
            x_f,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x_f.device),
            center=True
        )

        # Reshape back to (B, C, L)
        waveform = waveform.view(batch_size, num_channels, -1)

        # If it was originally a single channel (enhanced output), squeeze it
        if num_channels == 1:
            waveform = waveform.squeeze(1)

        return waveform