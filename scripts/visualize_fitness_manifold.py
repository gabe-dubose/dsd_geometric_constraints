#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

# --- parameters ---
SEED = 42
SIGMA = 1.0
RHO = 0.55
K_VALUES = [2, 8, 32]
N_SAMPLES = 16000

# --- model functions ---
def sample_g(K, n, rng):
    return rng.uniform(-1.0, 1.0, size=(n, K))

def q_orthogonal(g):
    return np.sum(g**2, axis=1)

def q_latent(g):
    K = g.shape[1]
    w1 = np.ones(K) / np.sqrt(K)
    w2 = np.array([1 if i % 2 == 0 else -1 for i in range(K)], dtype=float) / np.sqrt(K)
    t1 = g @ w1
    t2 = g @ w2
    return t1**2 + t2**2, t1, t2

def fitness(q, sigma):
    return np.exp(-q / (2.0 * sigma**2))

# --- colors: full viridis by K ---
viridis = mpl.colormaps['viridis']
viridis_positions = np.linspace(0.02, 0.98, len(K_VALUES))
K_colors = {K: np.array(viridis(pos)) for K, pos in zip(K_VALUES, viridis_positions)}

def blend_with_white(color, amount):
    white = np.array([1.0, 1.0, 1.0, 1.0])
    return tuple(np.array(color) * (1.0 - amount) + white * amount)

# --- plot ---
sns.set_style('whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

rng = np.random.default_rng(SEED)

for ci, K in enumerate(K_VALUES):
    g = sample_g(K, N_SAMPLES, rng)
    base_color = K_colors[K]
    rejected_color = blend_with_white(base_color, 0.68)
    accepted_color = blend_with_white(base_color, 0.08)

    # orthogonal expansion (top row)
    q1 = q_orthogonal(g)
    W1 = fitness(q1, SIGMA)
    acc = W1 >= RHO
    ax = axes[0][ci]
    ax.scatter(g[~acc, 0], g[~acc, 1], s=3.2, color=rejected_color, edgecolors='none', rasterized=True)
    ax.scatter(g[acc, 0],  g[acc, 1],  s=4.8, color=accepted_color, edgecolors='none', rasterized=True)
    ax.set_title(rf'$K={K}$', fontweight='bold')
    ax.set_xlabel(r'$g_1$')
    ax.set_ylabel(r'$g_2$' if ci == 0 else '')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect('equal', adjustable='box')

    # latent collapse (bottom row)
    q2, t1, t2 = q_latent(g)
    W2 = fitness(q2, SIGMA)
    acc2 = W2 >= RHO
    ax = axes[1][ci]
    ax.scatter(t1[~acc2], t2[~acc2], s=3.2, color=rejected_color, edgecolors='none', rasterized=True)
    ax.scatter(t1[acc2],  t2[acc2],  s=4.8, color=accepted_color, edgecolors='none', rasterized=True)
    ax.set_xlabel(r'$T_{K,1}(g)$')
    ax.set_ylabel(r'$T_{K,2}(g)$' if ci == 0 else '')
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect('equal', adjustable='box')

# shared legend on rightmost plots
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555',
           markersize=7, label=rf'High fitness ($W \geq {RHO}$)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#D0D0D0',
           markersize=7, label=rf'Low fitness ($W < {RHO}$)'),
]
for row in range(2):
    axes[row][2].legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)

# row labels
fig.text(0.01, 0.95, 'A', va='center', fontsize=14)
fig.text(0.01, 0.73, 'Orthogonal Expansion', rotation=90, va='center', fontsize=13)
fig.text(0.01, 0.475, 'B', va='center', fontsize=14)
fig.text(0.01, 0.27, 'Latent Collapse', rotation=90, va='center', fontsize=13)

out_dir = Path(__file__).resolve().parents[1] / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(out_dir / 'fitness_manifolds_geometry.png', dpi=300, bbox_inches='tight')
