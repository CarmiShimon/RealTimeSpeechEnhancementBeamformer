import torch
import torch.nn.functional as F


def wSDRLoss(clean, est, mix, eps=1e-8):
    """
    Standard Weighted SDR Loss (Time Domain).

    Args:
        clean: (B, L) Ground truth speech
        est:   (B, L) Enhanced speech
        mix:   (B, L) Original noisy mixture
    """

    def mSDR(a, b):
        # Dot product over the time dimension (L)
        numerator = torch.sum(a * b, dim=-1)
        # Product of magnitudes
        denominator = torch.norm(a, p=2, dim=-1) * torch.norm(b, p=2, dim=-1) + eps
        # Return negative correlation (mean across batch)
        return -torch.mean(numerator / denominator)

    # 1. Separate the actual noise from the mixture
    noise = mix - clean
    # 2. Separate the residual noise from the estimate
    noise_est = mix - est

    # 3. Calculate Alpha PER SAMPLE (B, 1) instead of per batch
    # Energy of clean speech vs energy of noise
    clean_energy = torch.sum(clean ** 2, dim=-1, keepdim=True)
    noise_energy = torch.sum(noise ** 2, dim=-1, keepdim=True)

    # Weighting factor based on SNR
    alpha = clean_energy / (clean_energy + noise_energy + eps)

    # 4. Compute weighted components
    speech_loss = mSDR(clean, est)
    noise_loss = mSDR(noise, noise_est)

    # We use mean of alpha for the final scalar loss
    return torch.mean(alpha) * speech_loss + torch.mean(1 - alpha) * noise_loss


def compute_CRM_loss(Clean_hat, Clean):
    loss_stft_real = F.mse_loss(Clean_hat.real, Clean.real)
    loss_stft_imag = F.mse_loss(Clean_hat.imag, Clean.imag)
    loss_stft = loss_stft_real + loss_stft_imag
    return loss_stft


def compute_IRM_loss(pred_mask, gt_mask):
    loss = F.mse_loss(pred_mask, gt_mask)
    return loss