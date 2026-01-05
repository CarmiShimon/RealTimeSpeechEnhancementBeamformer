import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import OmegaConf
import librosa
import soundfile as sf
import resampy
# Core components
from model import SpatialMaskNet
from stft_handler import STFTHandler
from beamformer import MVDRBeamformer
from norm import OnlineNorm
from targets import apply_bounded_crm


@torch.no_grad()
def run_full_inference(cfg_path):
    # 1. Configuration & Device
    cfg = OmegaConf.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Initialize Model
    model = SpatialMaskNet(cfg, device=device).to(device)  # Pass device to init
    checkpoint = torch.load(cfg.inference.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['loss']
    print(f'Loaded model with last val loss = {val_loss} and epoch = {epoch}')
    model.eval()

    # 3. Setup Processing Tools
    stft_handler = STFTHandler(cfg)
    beamformer = MVDRBeamformer(cfg)
    norm = OnlineNorm(cfg).to(device)
    decFac = cfg.model.get('decimation', 1)

    # 4. Audio Loading
    audio_front, fs = librosa.load(cfg.inference.wav_path_front, mono=False, sr=None)
    audio_rear, fs = librosa.load(cfg.inference.wav_path_rear, mono=False, sr=None)
    audio = np.concatenate((np.expand_dims(audio_front, 1), np.expand_dims(audio_rear, 1)), axis=-1)
    if fs != cfg.audio.sample_rate:
        audio = resampy.resample(audio.T, fs, cfg.audio.sample_rate)
        audio = torch.tensor(audio)
    if audio.shape[0] < 2:
        raise ValueError("Inference requires at least 2 channels for MVDR beamforming.")

    mix_t = audio.unsqueeze(0).to(device)  # [1, C, L]

    # 5. Domain Transformation
    mix_f = stft_handler.transform(mix_t)  # [1, C, T, F]

    # 6. Feature Extraction (LAS + GCC)
    mag_ref = torch.abs(mix_f[:, 0])
    las = torch.clamp(torch.log2(mag_ref + 1e-12), min=-100)

    # GCC Calculation
    cpsd = mix_f[:, 0] * torch.conj(mix_f[:, 1])
    cpsd_sliced = cpsd[:, :, 1:]

    gcc_complex = cpsd_sliced / (torch.abs(cpsd_sliced) + 1e-8)
    gcc = torch.cat((gcc_complex.real, gcc_complex.imag), dim=-1)  # GCC-PHAT

    # Temporal Decimation
    if decFac > 1:
        las = las[:, ::decFac, :]
        gcc = gcc[:, ::decFac, :]

    # Causal Normalization
    las_norm = norm(las, mode='stream')
    features = torch.cat((las_norm, gcc), dim=-1).float()

    # 7. RNN Inference
    h = model.init_hidden(batch_size=1)  # init_hidden uses self.device
    pred_mask_dec, _ = model(features, h)
    if cfg.mvdr == 'CRM':
        pred_real_mask = pred_mask_dec[..., 0]
        pred_imag_mask = pred_mask_dec[..., 1]
        pred_mask_dec = apply_bounded_crm(pred_real_mask, pred_imag_mask)
    # 8. Align predicted mask with STFT dimensions
    # Upsample Time
    if decFac > 1:
        pred_mask = torch.repeat_interleave(pred_mask_dec, decFac, dim=1)
        pred_mask = pred_mask[:, :mix_f.shape[2], :]
    else:
        pred_mask = pred_mask_dec

    # 9. Beamforming
    enhanced_f = beamformer(mix_f, pred_mask, mask_type=cfg.mvdr.mask_type)

    # 10. Reconstruction
    enhanced_t = stft_handler.inverse(enhanced_f).squeeze(0).cpu()

    # 11. Plotting and Export
    sf.write("../wavs/enhanced_output.wav", enhanced_t.unsqueeze(0).T, cfg.audio.sample_rate)
    plot_with_plotly(audio[0].cpu().numpy(), enhanced_t.numpy(), cfg.audio.sample_rate)


def plot_with_plotly(noisy, enhanced, fs):
    """Generates an interactive dashboard for audio comparison."""
    t = np.linspace(0, len(noisy) / fs, len(noisy))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Noisy Waveform", "Enhanced Waveform",
                        "Noisy Spectrogram", "Enhanced Spectrogram"),
        vertical_spacing=0.15
    )

    # Waveforms
    fig.add_trace(go.Scatter(x=t, y=noisy, name="Noisy", line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=enhanced, name="Enhanced", line=dict(color='blue')), row=1, col=2)

    # Spectrogram Helper
    def get_spec(sig, fs):
        x = torch.from_numpy(sig).float()

        Zxx = torch.stft(
            x,
            n_fft=512,
            hop_length=256,
            window=torch.hann_window(512),
            return_complex=True,
            center=False
        )

        spec_db = 20 * torch.log10(torch.abs(Zxx) + 1e-8)

        # Create axes
        T = Zxx.shape[1]
        F = Zxx.shape[0]
        t_spec = np.linspace(0, len(sig) / fs, T)
        f = np.linspace(0, fs / 2, F)

        return t_spec, f, spec_db.cpu().numpy()

    t_n, f_n, z_n = get_spec(noisy, fs)
    t_e, f_e, z_e = get_spec(enhanced, fs)

    # Heatmaps
    fig.add_trace(go.Heatmap(x=t_n, y=f_n, z=z_n, coloraxis="coloraxis"), row=2, col=1)
    fig.add_trace(go.Heatmap(x=t_e, y=f_e, z=z_e, coloraxis="coloraxis"), row=2, col=2)

    fig.update_layout(
        title="Inference Results: Noisy vs Enhanced",
        height=800,
        coloraxis=dict(colorscale='Viridis', cmin=-80, cmax=0),
        template="plotly_dark"
    )

    fig.show()


if __name__ == "__main__":
    run_full_inference("../conf/config.yaml")