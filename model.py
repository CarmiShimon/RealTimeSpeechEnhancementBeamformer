import torch
import torch.nn as nn
from omegaconf import DictConfig


class SpatialMaskNet(nn.Module):
    def __init__(self, cfg: DictConfig, device):
        super().__init__()
        self.device = device

        # 1. Dimensions
        self.bins = cfg.audio.n_fft // 2 + 1
        self.num_mics = cfg.audio.get('num_channels', 2)

        # Input: LAS (1) + GCC (2 per slave mic)
        self.input_dim = self.bins * (1 + 2 * (self.num_mics - 1)) - 2  # -1 for DC GCC DC band which has no spatial info
        self.hidden_size = cfg.model.hidden_dim
        self.num_layers = cfg.model.get('num_layers', 2)
        self.rnn_type = cfg.model.get('rnn_type', 'lstm').lower()

        # 2. RNN Selection
        if self.rnn_type == 'lstm':
            self.RNN = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False
            )
        else:
            # FIX: Use nn.GRU, not self.gru
            self.RNN = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False
            )

        # 3. Output Projection
        self.mask_type = cfg.mvdr.mask_type
        if self.mask_type == "CRM":
            self.fc = nn.Linear(self.hidden_size, self.bins * 2)
        else:
            self.fc = nn.Linear(self.hidden_size, self.bins)

        self.apply(self._init_weights)
        self.tanh = nn.Tanh()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, h=None):
        """
        Args:
            x: (B, T, D)
            h: Hidden state (h_0, c_0) for LSTM or h_0 for GRU
        """
        # FIX: Ensure h is passed and returned
        x, h_new = self.RNN(x, h)

        out = self.fc(x)

        if self.mask_type == "CRM":
            # Reshape into [B, T, F, 2] to get Real/Imag
            out = out.view(out.shape[0], out.shape[1], self.bins, 2)
            out = self.tanh(out)  # torch.complex(out[..., 0], out[..., 1])
        else:
            out = torch.sigmoid(out)

        return out, h_new

    def init_hidden(self, batch_size=1):
        """Initializes hidden state for LSTM or GRU."""
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            # LSTM needs a tuple (h, c)
            h = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
            c = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
            return h, c
        else:
            # GRU needs a single tensor h
            return weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)

