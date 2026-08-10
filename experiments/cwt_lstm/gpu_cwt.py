"""
GPU-accelerated Continuous Wavelet Transform via batched torch.conv1d.

pywt.cwt() (used in the first version of data.py) runs one window at a time
on CPU (~14ms/window measured), which makes the test-set CWT pass alone take
~20 minutes. But a CWT is fundamentally just "convolve the signal with a
bank of scaled wavelet kernels" -- exactly what Conv1d already does, and
Conv1d batches trivially across thousands of windows at once on a GPU. This
module reimplements the same complex-Morlet-style CWT directly as a
PyTorch op so it runs as one batched GPU convolution per scale instead of a
Python loop per window.

This is a different (from-scratch) wavelet kernel construction than pywt's
internal one, so outputs won't numerically match pywt's `cmor1.5-1.0`
exactly -- but it's the same wavelet family (complex Morlet) and the same
underlying math, and this project trains/evaluates entirely on its own
output, so internal consistency (not matching pywt bit-for-bit) is what
matters.

Wavelet: standard complex Morlet, psi(t) = pi^-0.25 * exp(i*w0*t) * exp(-t^2/2),
w0=6 (the common choice that satisfies the admissibility condition and gives
a good time-frequency resolution trade-off; the same w0 used by scipy's own
morlet2 default).
"""

import numpy as np
import torch
import torch.nn.functional as F

W0 = 6.0  # Morlet center angular frequency


def scale_for_frequency(freq_hz, fs, w0=W0):
    """Scale (in SAMPLES) whose dominant frequency is freq_hz, given sampling rate fs."""
    return w0 * fs / (2 * np.pi * freq_hz)


def build_morlet_kernels(scales, device="cpu", dtype=torch.float32, w0=W0):
    """
    Returns (real_kernels, imag_kernels): lists of 1D tensors, one pair per
    scale, each kernel covering +/-4 scale-widths (enough for the Gaussian
    envelope to decay to near-zero), odd-length so it centers exactly on the
    convolution's output sample.
    """
    real_kernels, imag_kernels = [], []
    for s in scales:
        half = max(1, int(np.ceil(4 * s)))
        m = 2 * half + 1
        t = (np.arange(m) - half) / s
        psi = (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2)
        psi = psi / np.sqrt(s)
        real_kernels.append(torch.tensor(psi.real, dtype=dtype, device=device))
        imag_kernels.append(torch.tensor(psi.imag, dtype=dtype, device=device))
    return real_kernels, imag_kernels


class GPUCWT:
    """Batched CWT: transforms (B, C, T) -> (B, C, num_scales, T) magnitude scalograms."""

    def __init__(self, scales, fs, device="cuda", w0=W0):
        self.scales = scales
        self.fs = fs
        self.device = device
        self.real_kernels, self.imag_kernels = build_morlet_kernels(scales, device=device, w0=w0)

    def transform(self, x):
        """x: (B, C, T) tensor, already on self.device. Returns (B, C, num_scales, T)."""
        B, C, T = x.shape
        x_flat = x.reshape(B * C, 1, T)
        mags = []
        for kr, ki in zip(self.real_kernels, self.imag_kernels):
            k_len = kr.shape[0]
            pad = k_len // 2
            # conv1d computes correlation; flip the kernel for a true convolution.
            kr_ = kr.flip(0).view(1, 1, -1)
            ki_ = ki.flip(0).view(1, 1, -1)
            real_part = F.conv1d(x_flat, kr_, padding=pad)[..., :T]
            imag_part = F.conv1d(x_flat, ki_, padding=pad)[..., :T]
            mags.append(torch.sqrt(real_part ** 2 + imag_part ** 2))
        out = torch.stack(mags, dim=1)          # (B*C, num_scales, T)
        return out.view(B, C, len(self.scales), T)

    def transform_numpy(self, x_np):
        """Convenience: x_np is (B, C, T) numpy array -> returns numpy (B, C, num_scales, T)."""
        x = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self.transform(x)
        return out.cpu().numpy()
