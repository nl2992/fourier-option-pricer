from __future__ import annotations
import numpy as np


def frft(x: np.ndarray, zeta: float) -> np.ndarray:
    """Fractional discrete Fourier transform via Bluestein's chirp-z decomposition.

    Computes
        G[n] = sum_{j=0}^{N-1} x[j] * exp(-i*2*pi*j*n*zeta),    n = 0, ..., N-1
    for arbitrary real zeta (standard FFT is the zeta = 1/N special case).

    Uses the identity  2*j*n = j^2 + n^2 - (j-n)^2  to turn the sum into a
    convolution evaluated with three length-2N FFTs:
        y_j = x_j * exp(-i*pi*zeta*j^2),          j = 0, ..., N-1,  zero-padded to 2N
        z_j = exp(i*pi*zeta*j^2) (length N), mirrored to length 2N around index N
        G_n = exp(-i*pi*zeta*n^2) * (y * z)_n,    n = 0, ..., N-1
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N < 1:
        raise ValueError("input must have length >= 1")

    j = np.arange(N)
    chirp = np.exp(1j * np.pi * zeta * j * j)             # e^{+i pi zeta j^2}

    # y: x * conj(chirp), padded with zeros to length 2N
    y = np.zeros(2 * N, dtype=np.complex128)
    y[:N] = x * np.conj(chirp)

    # z: [chirp_0, ..., chirp_{N-1}, 0, chirp_{N-1}, ..., chirp_1]  (length 2N)
    # so that circular conv gives (y * z)[n] = sum_j y[j] * chirp_{(n-j) mod 2N},
    # with chirp_{-k} = chirp_k since the chirp is a quadratic in |k|.
    z = np.zeros(2 * N, dtype=np.complex128)
    z[:N] = chirp
    if N > 1:
        z[N + 1:] = chirp[1:][::-1]

    conv = np.fft.ifft(np.fft.fft(y) * np.fft.fft(z))[:N]
    return np.conj(chirp) * conv
