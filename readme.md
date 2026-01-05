# Real-Time Multi-Channel Speech Enhancement & Separation

This repository implements a **causal, real-time capable multi-microphone speech enhancement pipeline** based on **SpatialMaskNet** and an **MVDR beamformer**.
The system is designed for low-latency streaming scenarios and supports both magnitude and complex masking strategies.

---

## 🚀 Features

* **Multi-Microphone Input**
  Uses **Log-Abs Spectrogram (LAS)** and **Generalized Cross-Correlation - PHAT(GCC)** spatial features.

* **Hybrid Training Objective**
  CRM mask prediction optimized using MSE loss
  Reconstructed signal is optimized using wSDR loss
  
* **Causal Architecture**
  Frame-by-frame processing with optional temporal decimation for real-time deployment.

* **Flexible Masking**
  Supports both:

  * **IRM** – Ideal Ratio Mask (real-valued magnitude mask)
  * **CRM** – Bounded Complex Ratio Mask (real + imaginary components)

* **Interactive Inference**
  Inference results can be visualized using interactive **Plotly dashboards**.

---

## 📁 Project Structure

* **`model.py`**
  SpatialMaskNet architecture (GRU/LSTM) with explicit hidden-state handling.

* **`stft_handler.py`**
  Forward and inverse STFT utilities for streaming and offline processing.

* **`beamformer.py`**
  MVDR beamformer implementation using predicted speech and noise masks.

* **`norm.py`**
  Online causal normalization for spectral features.

* **`trainer.py`**
  Training loop with epoch-level logic and hidden-state detachment.

* **`inference.py`**
  Script for enhancing WAV files and visualizing results.

---

## 🛠️ Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration Parameters

All system behavior is controlled via a YAML configuration file using **OmegaConf**.
Below is a detailed explanation of each configuration group.

---

### **`audio` — Signal Processing Parameters**

Controls STFT parameters and microphone configuration.

* **`sample_rate`**
  Sampling rate of the input audio in Hz. All WAV files must match this rate.

* **`n_fft`**
  FFT size for STFT. Frequency bins are computed as `n_fft // 2 + 1`.

* **`hop_length`**
  Number of samples between adjacent STFT frames. Controls latency and time resolution.

* **`num_mics`**
  Number of microphone channels used by the system.

---

### **`data` — Dataset and Segmentation**

Defines dataset locations and audio chunking strategy.

* **`train_path` / `val_path` / `test_path`**
  Paths to HDF5 datasets containing multichannel audio and targets.

* **`segment_sec`**
  Duration (seconds) of audio segments used during training.

* **`stride_sec`**
  Temporal stride between consecutive segments. `0` means no overlap.

---

### **`mvdr` — Beamforming and Mask Control**

Controls MVDR beamformer behavior and mask interpretation.

* **`mask_type`**
  Mask representation predicted by the network:

  * `IRM` – magnitude-only mask
  * `CRM` – complex-valued mask (real + imaginary)

* **`is_use_phase`**
  If enabled, normalizes the Relative Transfer Function (RTF) to unit magnitude and relies only on phase differences.

* **`noise_thd`**
  Threshold for selecting noise-dominant time–frequency bins.

* **`speech_thd`**
  Threshold for selecting speech-dominant time–frequency bins.

* **`Tsmth`**
  Time constant (seconds) for exponential smoothing of PSD estimates.

---

### **`training` — Optimization Settings**

Defines how the model is trained.

* **`epochs`**
  Number of training epochs.

* **`batch_size`**
  Number of samples per batch.

* **`lr`**
  Learning rate for the optimizer.

* **`weight_decay`**
  L2 regularization coefficient.

* **`num_workers`**
  Number of parallel workers for data loading.

* **`checkpoint_file`**
  Output directory for checkpoints and logs.

---

### **`model` — Neural Network Architecture**

Controls SpatialMaskNet architecture.

* **`rnn_type`**
  Recurrent layer type: `gru` or `lstm`.

* **`hidden_dim`**
  Hidden-state dimensionality of the RNN.

* **`num_hidden_layers`**
  Number of stacked RNN layers.

* **`decimation`**
  Temporal downsampling factor. Higher values reduce computation and latency at the cost of temporal resolution.

* **`use_gcc`**
  Enables GCC spatial features in addition to LAS.

---

### **`norm` — Online Normalization**

Controls causal normalization of spectral features.

* **`Tpsd`**
  Time constant (seconds) for PSD smoothing used during online normalization.

> Normalization is applied to LAS features only. Feature dimensionality is derived automatically from `n_fft`.

---

### **`inference` — Runtime Evaluation**

Parameters used during inference and visualization.

* **`model_path`**
  Path to the trained model checkpoint (`.pt`).

* **`wav_path_front`**
  Path to the front microphone WAV file.

* **`wav_path_rear`**
  Path to the rear microphone WAV file.

> Input WAV files must be time-synchronized and sampled at `audio.sample_rate`.

---

## 📝 Notes

* All components are designed to be **causal and streaming-safe**.
* Configuration changes do **not** require code modification.
* The architecture is suitable for real-time CPU inference and can be quantized for deployment.

---


Specify license information here.



