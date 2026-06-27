"""
Visualize and compare HiPPO Legendre basis functions vs Daubechies wavelet basis.

Goal: check whether these two bases are complementary (orthogonal) to each other.
If their inner products are near zero, they capture different spatial patterns
and combining them in two parallel SSM streams would be non-redundant.

Run from models/yolov9/:
    python visualize_basis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.linalg import expm


# --- step 1: compute HiPPO Legendre basis functions ---
# HiPPO-LegS A matrix (Legendre scaled) -- this is what S4/S4ND uses by default.
# the hidden state h(t) tracks projections of the input onto Legendre polynomials.
# we compute the impulse response of each mode to get the actual basis functions.

def hippo_legs_AB(N):
    # from the HiPPO paper: Legendre (scaled) measure
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)[:, None]
    return A, B


def compute_legendre_basis(N, L):
    """
    Compute the impulse response of each HiPPO Legendre mode.
    Returns phi of shape (N, L) -- each row is one basis function over L timesteps.
    The impulse response shows what spatial pattern each mode tracks.
    """
    A, B = hippo_legs_AB(N)

    # discretize with a small dt so we get L steps
    dt = 1.0 / L
    # ZOH discretization: A_bar = exp(dt * A), B_bar = (A_bar - I) A^{-1} B
    A_bar = expm(dt * A)
    # simple Euler for B_bar (close enough for visualization)
    B_bar = dt * B

    # run the SSM with a unit impulse at t=0
    h = np.zeros(N)
    phi = np.zeros((N, L))
    for t in range(L):
        if t == 0:
            h = A_bar @ h + B_bar[:, 0] * 1.0   # impulse at t=0
        else:
            h = A_bar @ h                         # no input after t=0
        phi[:, t] = h

    return phi   # (N, L)


# --- step 2: compute Daubechies wavelet basis functions ---
# we use pywt to get the wavelet filter coefficients and then build
# the basis functions at the same resolution L as the Legendre basis.

def compute_daubechies_basis(wavelet_name, L):
    """
    Compute Daubechies basis functions at resolution L.
    Returns psi of shape (M, L) -- each row is one wavelet at a different
    scale/translation, resampled to length L.
    """
    wavelet = pywt.Wavelet(wavelet_name)

    # get the wavelet function at high resolution then downsample to L
    # wavelet.wavefun returns (phi, psi, x) for orthogonal wavelets
    # phi = scaling function, psi = wavelet function
    phi_wav, psi_wav, x = wavelet.wavefun(level=10)

    # resample both to length L
    from scipy.interpolate import interp1d
    x_new = np.linspace(x[0], x[-1], L)
    psi_resampled = interp1d(x, psi_wav, kind='linear', fill_value=0, bounds_error=False)(x_new)
    phi_resampled = interp1d(x, phi_wav, kind='linear', fill_value=0, bounds_error=False)(x_new)

    # build a small set of wavelets at different scales by stretching/shifting
    # scale s=1 (original), s=2 (stretched x2), s=4 (stretched x4)
    basis = []
    for scale in [1, 2, 4, 8]:
        x_scaled = np.linspace(x[0] * scale, x[-1] * scale, L)
        psi_scaled = interp1d(
            x * scale, psi_wav,
            kind='linear', fill_value=0, bounds_error=False
        )(x_scaled)
        # normalize
        norm = np.linalg.norm(psi_scaled)
        if norm > 1e-8:
            psi_scaled /= norm
        basis.append(psi_scaled)

    # also include the scaling function (low-pass complement)
    norm = np.linalg.norm(phi_resampled)
    if norm > 1e-8:
        phi_resampled /= norm
    basis.append(phi_resampled)

    return np.stack(basis, axis=0)   # (M, L)


# --- step 3: compute cross-correlation (inner products) ---
# if Legendre and Daubechies bases are orthogonal, the inner product matrix
# should be near zero everywhere. high values mean overlap / redundancy.

def cross_gram_matrix(basis_a, basis_b):
    """
    Compute the Gram matrix G[i,j] = <basis_a[i], basis_b[j]> / (||a_i|| * ||b_j||).
    Values close to 0 mean orthogonal (complementary).
    Values close to 1 mean parallel (redundant).
    """
    # normalize each row
    a = basis_a / (np.linalg.norm(basis_a, axis=1, keepdims=True) + 1e-8)
    b = basis_b / (np.linalg.norm(basis_b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T   # (N, M)


# --- main ---

N = 64    # d_state=64 as used in S4ND
L = 512   # sequence length for visualization

print("computing HiPPO Legendre basis (N=64)...")
legendre_basis = compute_legendre_basis(N, L)   # (64, L)

# compare wavelets with increasing vanishing moments to show the tension
wavelets_to_test = ['sym4', 'sym8', 'sym20', 'db38']

print("computing wavelet bases...")
wavelet_bases = {}
for wname in wavelets_to_test:
    wavelet_bases[wname] = compute_daubechies_basis(wname, L)

grams = {}
for wname, basis in wavelet_bases.items():
    grams[wname] = cross_gram_matrix(legendre_basis, basis)
    print(f"Legendre(N=64) vs {wname:6s} -- max |inner product|: {np.abs(grams[wname]).max():.4f}  mean: {np.abs(grams[wname]).mean():.4f}")

print("(values near 0 = complementary, values near 1 = redundant)")

# --- plot ---

fig = plt.figure(figsize=(16, 18))
t = np.linspace(0, 1, L)

# plot 1: first 8 Legendre basis functions
ax1 = fig.add_subplot(4, 2, (1, 2))
for i in range(8):
    ax1.plot(t, legendre_basis[i], label=f'mode {i}', alpha=0.7)
ax1.set_title('HiPPO Legendre basis (N=64, showing first 8 modes)\nGlobal, smooth — spread across entire domain', fontsize=11)
ax1.set_xlabel('position')
ax1.set_ylabel('amplitude')
ax1.legend(loc='upper right', fontsize=7, ncol=4)
ax1.axhline(0, color='k', linewidth=0.5)
ax1.grid(True, alpha=0.3)

# plot 2: wavelet basis functions for each variant — show the localization trade-off
wavelet_colors = ['sym4', 'sym8', 'sym20', 'db38']
titles = {
    'sym4':  'sym4  (4 vanishing moments) — very localized',
    'sym8':  'sym8  (8 vanishing moments) — moderately localized',
    'sym20': 'sym20 (20 vanishing moments) — losing localization',
    'db38':  'db38  (38 vanishing moments) — nearly global, max available',
}
for idx, wname in enumerate(wavelets_to_test):
    ax = fig.add_subplot(4, 2, 3 + idx)
    basis = wavelet_bases[wname]
    for i in range(len(basis)):
        ax.plot(t, basis[i], alpha=0.8)
    g = grams[wname]
    ax.set_title(f'{titles[wname]}\nmax overlap={np.abs(g).max():.3f}  mean overlap={np.abs(g).mean():.3f}', fontsize=9)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('position')
    ax.set_ylabel('amplitude')
    ax.grid(True, alpha=0.3)

# plot 3: summary — overlap vs vanishing moments (the tension)
ax_summary = fig.add_subplot(4, 2, (7, 8))
vms = [4, 8, 20, 38]
max_overlaps = [np.abs(grams[w]).max() for w in wavelets_to_test]
mean_overlaps = [np.abs(grams[w]).mean() for w in wavelets_to_test]
ax_summary.plot(vms, max_overlaps, 'o-', label='max overlap', color='red')
ax_summary.plot(vms, mean_overlaps, 's-', label='mean overlap', color='blue')
ax_summary.axvline(64, color='gray', linestyle='--', label='N=64 (d_state)')
ax_summary.set_xlabel('vanishing moments')
ax_summary.set_ylabel('overlap with Legendre(N=64)')
ax_summary.set_title('The tension: more vanishing moments = more complementary to Legendre\nbut wavelet becomes less localized (wider support)', fontsize=10)
ax_summary.legend()
ax_summary.grid(True, alpha=0.3)
ax_summary.set_ylim(0, 1)

plt.tight_layout()
out_path = 'legendre_vs_wavelet_basis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nsaved plot to {out_path}")
plt.close()
