import torch
from omegaconf import DictConfig
from losses import compute_CRM_loss, compute_IRM_loss
from stft_handler import STFTHandler
from beamformer import MVDRBeamformer
from norm import OnlineNorm
from targets import apply_bounded_crm, get_irm_mask


class Trainer:
    def __init__(self, cfg: DictConfig, model, device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.eps = 1e-12
        self.batch_size = self.cfg.training.batch_size
        self.stft_handler = STFTHandler(cfg)
        self.beamformer = MVDRBeamformer(cfg)
        self.norm = OnlineNorm(cfg).to(device)

        self.decFac = cfg.model.get('decimation', 1)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )

    def extract_features(self, mix_f, mode='batch'):
        """
        Extracts and normalizes fused LAS + GCC features.
        mix_f: (B, C, T, F)
        """
        # 1. LAS (Log Absolute Spectrogram) - Reference Mic 0
        # mix_f is (B, C, T, F), we want (B, T, F)
        mag_ref = torch.abs(mix_f[:, 0])
        las = torch.log2(mag_ref + self.eps)
        las = torch.clamp(las, min=-100)

        # 2. GCC (Generalized Cross-Correlation)
        # ch0 * conj(ch1)
        # GCC for all slave mics: (C-1) pairs
        gcc_list = []
        for i in range(1, mix_f.shape[1]):
            cpsd = mix_f[:, 0] * torch.conj(mix_f[:, i])
            cpsd = cpsd[:, :, 1:]   # remove DC band
            gcc_pair = cpsd / (torch.abs(cpsd) + 1e-8)
            gcc_list.append(torch.cat((gcc_pair.real, gcc_pair.imag), dim=-1))

        gcc_all = torch.cat(gcc_list, dim=-1)  # [B, T, F * 2 * (C-1)]

        # 3. Decimation (Temporal Downsampling)
        if self.decFac > 1:
            las = las[:, ::self.decFac, :]
            gcc_all = gcc_all[:, ::self.decFac, :]

        # 4. Normalization (Applied only to LAS)
        las = self.norm(las, mode=mode)

        # 5. Fusion: (B, T_dec, F_las + F_gcc)
        return torch.cat((las, gcc_all), dim=-1).float()

    def train_epoch(self, train_loader, epoch):
        """
        Processes one full epoch of training with RNN state management and Hybrid Loss.
        """
        self.model.train()
        total_epoch_loss = 0

        # Initialize hidden state for the start of the epoch
        h = self.model.init_hidden(batch_size=self.cfg.training.batch_size)

        for batch_idx, batch in enumerate(train_loader):
            # 1. Handle remainder batch size
            current_batch_size = batch[0].size(0)
            if (isinstance(h, tuple) and h[0].size(1) != current_batch_size) or \
                    (not isinstance(h, tuple) and h.size(1) != current_batch_size):
                h = self.model.init_hidden(batch_size=current_batch_size)

            # 2. Unpack and Transform
            mix_t, clean_t = [b.to(self.device) for b in batch]
            mix_f = self.stft_handler.transform(mix_t)  # [B, C, T, F]
            clean_f_ref = self.stft_handler.transform(clean_t)[:, 0]  # [B, T, F] reference mic

            # 3. Features & Hidden State Management
            features = self.extract_features(mix_f, mode='batch')

            # Detach hidden state (TBPTT)
            if isinstance(h, tuple):
                h = tuple(state.detach() for state in h)
            else:
                h = h.detach()

            # 4. Forward Pass
            pred_mask_dec, h = self.model(features, h)
            if self.cfg.mvdr == 'CRM':
                pred_real_mask = pred_mask_dec[..., 0]
                pred_imag_mask = pred_mask_dec[..., 1]
                pred_mask_dec = apply_bounded_crm(pred_real_mask, pred_imag_mask)

            # 5. Temporal up-sampling (if decimated)
            if self.decFac > 1:
                pred_mask = torch.repeat_interleave(pred_mask_dec, self.decFac, dim=1)
                # Match STFT frames exactly
                if pred_mask.shape[1] < mix_f.shape[2]:
                    pred_mask = torch.nn.functional.pad(pred_mask, (0, 0, 0, mix_f.shape[2] - pred_mask.shape[1]),
                                                        mode='replicate')
                else:
                    pred_mask = pred_mask[:, :mix_f.shape[2], :]
            else:
                pred_mask = pred_mask_dec

            # 6. Loss Calculation
            with torch.no_grad():
                if self.cfg.mvdr.mask_type == "IRM":
                    gt_mask = get_irm_mask(clean_f_ref, mix_f)


            # 7. STFT loss
            # enhanced_f = self.beamformer(mix_f, pred_mask, mask_type=self.cfg.mvdr.mask_type)
            if self.cfg.mvdr.mask_type == "IRM":
                total_loss = compute_IRM_loss(pred_mask, gt_mask)
            else:
                clean_f_hat = torch.complex(pred_mask[..., 0], pred_mask[..., 1]) * mix_f[:, 0]
                total_loss = compute_CRM_loss(clean_f_hat, clean_f_ref)

            # 8. Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping is essential for RNN stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_epoch_loss += total_loss.item()

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.6f}")

        return total_epoch_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using 'stream' mode
        to simulate real-time performance.
        """
        self.model.eval()
        total_val_loss = 0.0
        # Initialize hidden state
        h = self.model.init_hidden(batch_size=self.cfg.training.batch_size)
        for batch_idx, batch in enumerate(val_loader):
            current_batch_size = batch[0].size(0)
            if (isinstance(h, tuple) and h[0].size(1) != current_batch_size) or \
                    (not isinstance(h, tuple) and h.size(1) != current_batch_size):
                h = self.model.init_hidden(batch_size=current_batch_size)
            # 1. Load and Transform
            mix_t, clean_t = [b.to(self.device) for b in batch]
            mix_f = self.stft_handler.transform(mix_t)
            clean_f_ref = self.stft_handler.transform(clean_t)[:, 0]

            # 3. Feature Extraction
            features = self.extract_features(mix_f, mode='batch')

            # 4. Model Inference

            # Detach hidden state (TBPTT)
            if isinstance(h, tuple):
                h = tuple(state.detach() for state in h)
            else:
                h = h.detach()
            pred_mask_dec, h = self.model(features, h)

            # 5. Temporal up-sampling (if decimated)
            if self.decFac > 1:
                pred_mask = torch.repeat_interleave(pred_mask_dec, self.decFac, dim=1)
                if pred_mask.shape[1] < mix_f.shape[2]:
                    pred_mask = torch.nn.functional.pad(
                        pred_mask, (0, 0, 0, mix_f.shape[2] - pred_mask.shape[1]), mode='replicate'
                    )
                else:
                    pred_mask = pred_mask[:, :mix_f.shape[2], :]
            else:
                pred_mask = pred_mask_dec

            # 6. Compute Loss
            if self.cfg.mvdr.mask_type == "IRM":
                gt_mask = get_irm_mask(clean_f_ref, mix_f)
                irm_loss = compute_IRM_loss(pred_mask, gt_mask)
                total_val_loss += irm_loss.item()
            else:
                clean_f_hat = torch.complex(pred_mask[..., 0], pred_mask[..., 1]) * mix_f[:, 0]
                crm_loss = compute_CRM_loss(clean_f_hat, clean_f_ref)
                total_val_loss += crm_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Finished | Avg Loss: {avg_val_loss:.4f}")

        return avg_val_loss
