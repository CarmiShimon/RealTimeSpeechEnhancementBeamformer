# Real-Time Multi-Channel Speech Enhancement&separation (SpatialMaskNet + MVDR)

This repository implements a causal, real-time capable speech enhancement pipeline using Spatial Features, GRU/LSTM and an MVDR Beamformer.

## 🚀 Features
* **Multi-Microphone Input:** Uses Log-Abs Spectrogram (LAS) and Generalized Cross-Correlation (GCC) features.
* **Hybrid Loss:** Combines Mask MSE and Signal-level loss (via MVDR) for superior intelligibility.
* **Causal Architecture:** Frame-by-frame processing with decimation support for low-latency applications.
* **Flexible Masking:** Supports both Ideal Ratio Mask (IRM) and Bounded Complex Ratio Mask (CRM).
* **Interactive Inference:** Visualizes results using Plotly dashboards.

## 📁 Project Structure
* `model.py`: The model architecture with hidden state management.
* `stft_handler.py`: Handles forward and inverse STFT.
* `beamformer.py`: Implementation of the MVDR beamformer using predicted masks.
* `norm.py`: Online causal normalization for streaming features.
* `trainer.py`: Epoch-level training logic with hidden state detachment.
* `inference.py`: Interactive script to enhance WAV files and plot results.

## 🛠️ Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt