#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch, Ellipse

BLACK = "#111111"

def add_phenotype(ax, xy):
    ellipse = Ellipse(xy, width=3.25, height=1.85, fill=False,
                      linewidth=1.7, edgecolor=BLACK)
    ax.add_patch(ellipse)
    ax.text(xy[0], xy[1], 'Phenotype', ha='center', va='center',
            fontsize=12, fontweight='bold')

def draw_orthogonal(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 10.65)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.65, 'A', fontsize=18, ha='left', va='top')
    ax.text(0.75, 9.65, 'Orthogonal Expansion', fontsize=14, ha='left', va='top')
    y_positions = [8.0, 6.65, 5.3, 3.95, 2.6]
    for i, y in enumerate(y_positions, start=1):
        ax.add_patch(Circle((1.3, y), 0.36, fill=False, linewidth=1.4, edgecolor=BLACK))
        ax.text(1.3, y, rf'$g_{{{i}}}$', ha='center', va='center', fontsize=12)
        ax.add_patch(FancyArrowPatch((1.72, y), (4.2, y), arrowstyle='->', mutation_scale=12, linewidth=1.2, color=BLACK))
    for i, y in enumerate(y_positions, start=1):
        ax.add_patch(Rectangle((4.25, y - 0.36), 1.18, 0.72, fill=False, linewidth=1.4, edgecolor=BLACK))
        ax.text(4.84, y, rf'$\phi_{{{i}}}$', ha='center', va='center', fontsize=12)
        ax.add_patch(FancyArrowPatch((5.48, y), (6.98, 5.3), arrowstyle='->', mutation_scale=12, linewidth=1.15, color=BLACK))
    add_phenotype(ax, (8.65, 5.3))

def draw_latent(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 10.65)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.65, 'B', fontsize=18, ha='left', va='top')
    ax.text(0.75, 9.65, 'Latent Collapse', fontsize=14, ha='left', va='top')
    y_positions = [8.05, 7.0, 5.95, 4.9, 3.85, 2.8, 1.75]
    for i, y in enumerate(y_positions, start=1):
        ax.add_patch(Circle((1.25, y), 0.32, fill=False, linewidth=1.4, edgecolor=BLACK))
        label = rf'$g_{{{i}}}$' if i < 7 else r'$\cdots$'
        ax.text(1.25, y, label, ha='center', va='center', fontsize=11)
        ax.add_patch(FancyArrowPatch((1.62, y), (4.4, 5.0), arrowstyle='->', mutation_scale=11, linewidth=1.0, color=BLACK, alpha=0.95))
    ax.add_patch(Polygon([[4.35, 7.35], [5.95, 6.35], [5.95, 3.65], [4.35, 2.65]],
                          fill=False, linewidth=1.7, edgecolor=BLACK))
    ax.text(5.15, 5.0, r'$T_K(g)$', ha='center', va='center', fontsize=13)
    ax.add_patch(FancyArrowPatch((6.05, 5.0), (6.92, 5.0), arrowstyle='->', mutation_scale=12, linewidth=1.2, color=BLACK))
    add_phenotype(ax, (8.72, 5.0))

# --- plot ---
sns.set_style('white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.1))
draw_orthogonal(ax1)
draw_latent(ax2)
plt.subplots_adjust(wspace=0.1, left=0.025, right=0.99, top=0.98, bottom=0.04)
out_dir = Path(__file__).resolve().parents[1] / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'architecture_diagram.png', dpi=300, bbox_inches='tight')
