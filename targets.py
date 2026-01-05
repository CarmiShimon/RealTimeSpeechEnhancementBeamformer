import torch

eps = 1e-12

def apply_bounded_crm(real, imag):
    # real, imag: [B, T, F]
    # mixture: [B, F, T] complex STFT

    # Permute for alignment: [B, T, F] -> [B, F, T]
    real, imag = real.permute(0, 2, 1), imag.permute(0, 2, 1)

    # Form complex mask
    M = torch.complex(real, imag)

    # Compute magnitude
    mag = torch.abs(M)
    mag_mask = torch.tanh(mag)  # bounded in [0, 1]

    # Phase normalization
    phase = M / (mag + 1e-8)

    # Final complex mask
    mask = mag_mask * phase  # [B, F, T]

    # Apply to mixture
    # enhanced = mask * mixture  # [B, F, T]
    return mask

def get_irm_mask(clean_f_ref, mix_f):
    clean_mag = torch.abs(clean_f_ref)
    noise_mag = torch.abs(mix_f[:, 0] - clean_f_ref)
    gt_mask = torch.sqrt(clean_mag.pow(2) / (clean_mag.pow(2) + noise_mag.pow(2) + eps))
    gt_mask = torch.clamp(gt_mask, 0.0, 1.0)
    return gt_mask