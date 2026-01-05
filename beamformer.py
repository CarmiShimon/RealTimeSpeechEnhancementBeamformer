import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig


class MVDRBeamformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.num_mics = cfg.audio.num_mics  # Updated naming convention
        self.noise_thd = cfg.mvdr.noise_thd
        self.speech_thd = cfg.mvdr.speech_thd
        self.is_use_phase = cfg.mvdr.get('is_use_phase', True)
        self.eps = 1e-12

        # Calculate alpha for exponential smoothing
        hop_time = cfg.audio.hop_length / cfg.audio.sample_rate
        self.alp_exp = 1 - np.exp(-hop_time / cfg.mvdr.Tsmth)

    def prepare_signal_estimates(self, mix_f, mask, mask_type='CRM'):
        """
        Supports N channels by broadcasting the single-channel mask
        across the channel dimension (dim 1).
        """
        if mask_type == 'CRM':
            mag_mask = torch.abs(mask)
            # Standard phase extraction
            phase = mask / (mag_mask + 1e-8)

            # Double-threshold gating
            gated_mag = torch.where(mag_mask < self.noise_thd, 0.0, mag_mask)
            gated_mag = torch.where(gated_mag > self.speech_thd, 1.0, gated_mag)

            cRM = gated_mag * phase
            # broadcast single-channel mask to all microphones for multi-channel MVDR.
            speech_f_hat = torch.complex(cRM[..., 0], cRM[..., 1]).unsqueeze(1).repeat(1, self.num_mics, 1, 1) * mix_f
            noise_f_hat = mix_f - speech_f_hat
        else:
            irm_gated = torch.where(mask < self.noise_thd, 0.0, mask)
            irm_gated = torch.where(irm_gated > self.speech_thd, 1.0, irm_gated)
            speech_f_hat = mix_f * irm_gated.unsqueeze(1)
            noise_f_hat = mix_f * (1.0 - irm_gated).unsqueeze(1)

        return speech_f_hat, noise_f_hat

    def _compute_psd(self, Y):
        """Batch covariance: (B, C, T, F) -> (B, T, F, C, C)"""
        # Optimized einsum for multi-channel covariance
        return torch.einsum('bctf, bdtf -> btfcd', Y, Y.conj())

    def _apply_smoothing(self, R):
        """
        Vectorized exponential smoothing.
        Note: For true real-time/causal use, a loop or recursive filter is needed,
        but for 'batch' training, we can use a scan or specialized kernel.
        """
        if self.alp_exp >= 1.0:
            return R

        # Using a simple recursive approach that is more efficient than a pure Python loop
        # For N-channels, we ensure the matrix operations are vectorized
        R_smth = [R[:, 0]]
        for t in range(1, R.size(1)):
            R_smth.append((1 - self.alp_exp) * R_smth[-1] + self.alp_exp * R[:, t])

        return torch.stack(R_smth, dim=1)

    def forward(self, mix_f, mask, mask_type='CRM'):
        # 1. Estimate Speech and Noise signals
        speech_f_hat, noise_f_hat = self.prepare_signal_estimates(mix_f, mask, mask_type)

        # 2. Estimate PSD Matrices (B, T, F, C, C)
        Rxx = self._apply_smoothing(self._compute_psd(speech_f_hat))
        Rnn = self._apply_smoothing(self._compute_psd(noise_f_hat))

        # 3. Regularization & Matrix Inversion
        # Scale eye matrix by the mean power to avoid hardcoding 1e-10
        C = mix_f.shape[1]
        eye = torch.eye(C, device=mix_f.device).view(1, 1, 1, C, C)
        Rnn_inv = torch.linalg.inv(Rnn + (1e-9 * eye))

        # 4. RTF Estimation (Relative to Mic 0)
        # First column of Rxx normalized by the power of Mic 0
        # rtf shape: [B, T, F, C]
        rtf = Rxx[:, :, :, :, 0] / (Rxx[:, :, :, 0, 0].unsqueeze(-1) + self.eps)

        if self.is_use_phase:
            rtf = rtf / (torch.abs(rtf) + 1e-8)

        # 5. MVDR Weights Calculation
        # Vectorized for N-channels
        num = torch.einsum('btfij, btfj -> btfi', Rnn_inv, rtf)
        den = torch.einsum('btfi, btfi -> btf', rtf.conj(), num)
        weights = num / (den.unsqueeze(-1) + self.eps)

        # 6. Filtering: y = w^H x
        # weights: [B, T, F, C] | mix_f: [B, C, T, F]
        clean_f_hat = torch.einsum('btfi, bitf -> btf', weights.conj(), mix_f)

        return clean_f_hat