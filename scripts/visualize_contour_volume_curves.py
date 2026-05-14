#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from matplotlib.lines import Line2D

# --- parameters ---
SEED = 42
SIGMA = 1.0
RHO_GRID = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
K_VALUES = [2, 4, 8, 16, 32]
N_SAMPLES = 70000

# --- model functions ---
def sample_g(K, n, rng):
    return rng.uniform(-1.0, 1.0, size=(n, K))

def q_orthogonal(g):
    return np.sum(g**2, axis=1)

def q_latent(g):
    K = g.shape[1]
    w1 = np.ones(K) / np.sqrt(K)
    w2 = np.array([1 if i % 2 == 0 else -1 for i in range(K)], dtype=float) / np.sqrt(K)
    return (g @ w1)**2 + (g @ w2)**2

def fitness(q, sigma):
    return np.exp(-q / (2.0 * sigma**2))

def volume_curve(W, rho_grid):
    return np.array([np.mean(W >= rho) for rho in rho_grid])

# --- colors: full viridis by K ---
viridis = mpl.colormaps['viridis']
viridis_positions = np.linspace(0.02, 0.98, len(K_VALUES))
K_colors = {K: viridis(pos) for K, pos in zip(K_VALUES, viridis_positions)}

# --- run simulation ---
rng = np.random.default_rng(SEED)
orth_curves = {}
lat_curves  = {}

for K in K_VALUES:
    g = sample_g(K, N_SAMPLES, rng)
    orth_curves[K] = volume_curve(fitness(q_orthogonal(g), SIGMA), RHO_GRID)
    lat_curves[K]  = volume_curve(fitness(q_latent(g), SIGMA), RHO_GRID)

# --- plot ---
sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for K in K_VALUES:
    ax1.plot(RHO_GRID, orth_curves[K], marker='o', linewidth=2.0, markersize=4.2,
             color=K_colors[K], label=rf'$K={K}$')
    ax2.plot(RHO_GRID, lat_curves[K], marker='o', linewidth=2.0, markersize=4.2,
             color=K_colors[K], label=rf'$K={K}$')

for ax, title in zip([ax1, ax2], ['Orthogonal Expansion', 'Latent Collapse']):
    ax.set_xlabel(r'Fitness contour ($\rho$)', fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, fontsize=9, title='Complexity')

ax1.set_ylabel(r'Contour volume $V_K(\rho)$', fontsize=12)
ax1.text(-0.1, 1.04, 'A', transform=ax1.transAxes, fontsize=14, va='top', ha='left')
ax2.text(-0.07, 1.04, 'B', transform=ax2.transAxes, fontsize=14, va='top', ha='left')

ax1.text(0.83, 0.62, s=f'Orthogonal\nExpansion', fontsize=10)
ax2.text(0.83, 0.62, s=f'Latent\nCollapse', fontsize=10)

plt.tight_layout()
out_dir = Path(__file__).resolve().parents[1] / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'volume_curves_split_viridis_byK.png', dpi=300, bbox_inches='tight')
