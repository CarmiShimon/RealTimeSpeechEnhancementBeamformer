import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from stft_handler import STFTHandler


class SpatialDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.h5_path = cfg.data.train_path if split == 'train' else cfg.data.val_path
        self.stft_handler = STFTHandler(cfg)

        # Initialize File Handler
        self.data_set = Audioset(
            h5_path=self.h5_path,
            length=self.cfg.data.segment_sec,
            stride=self.cfg.data.stride_sec,
            sample_rate=self.cfg.audio.sample_rate
        )

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        # 1. Get raw time-domain waveforms
        mix_t, clean_t = self.data_set[idx]
        return mix_t, clean_t


class Audioset:
    def __init__(self, h5_path=None, length=None, stride=None, sample_rate=None):
        self.h5_path = h5_path
        self.sample_rate = sample_rate
        self.chunk_len = length
        self.chunk_stride = stride
        self.length = int(self.chunk_len * self.sample_rate)

        # Metadata check for multiprocessing safety
        with h5py.File(self.h5_path, 'r') as f:
            mix_len = f['mix'].shape[1]
            # Use your specific length logic
            self.total_len = int(mix_len / self.length) - int(self.length / self.sample_rate)
        self.dataset = None

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if self.dataset is None:
            # swmr=True (Single-Writer-Multiple-Reader) is better for concurrent reading
            self.dataset = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        start = index * self.length
        end = (index + 1) * self.length
        mix = self.dataset['mix'][:, start:end].astype(np.float32)
        target = self.dataset['target'][:, start:end].astype(np.float32)
        # Basic cleanup and conversion to torch
        mix_t = torch.from_numpy(np.nan_to_num(mix, nan=0.0)) / 32768.0
        clean_t = torch.from_numpy(np.nan_to_num(target, nan=0.0)) / 32768.0

        return mix_t, clean_t