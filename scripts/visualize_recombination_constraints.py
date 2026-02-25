#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mendelian_data_file = '../data/mendelian_architecture_dsd_dynamics.json'
with open(mendelian_data_file, 'r') as infile:
    mendelian_data = json.load(infile)

mendelian_traits_file = '../data/mendelian_architecture_mean_trait_histories.json'
with open(mendelian_traits_file, 'r') as infile:
    mendelian_traits = json.load(infile)
    
# initialie plot
sns.set_style('whitegrid')
fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(12, 4))

# define color palette
colors = sns.color_palette("viridis", n_colors=5)

# ax1: mean trait trajectories
for g, recombination_rate in enumerate(sorted(mendelian_traits.keys(), key=float)):
    trajectory = np.array(mendelian_traits[recombination_rate])
    ax1.plot(trajectory[:, 0], trajectory[:, 1], color=colors[g], linewidth=2,
        marker='o', markersize=2, label=rf"$r$ = {recombination_rate}", alpha=0.75)

ax1.set_xlabel(r"Mean $a$", fontsize=12)
ax1.set_ylabel(r"Mean $b$", fontsize=12)
ax1.legend(title=r"Recomb. rate ($r$)")

# ax2: mendelian inheritance
for g, recombination_rate in enumerate(mendelian_data.keys()):
    data = mendelian_data[recombination_rate]
    data_array = np.vstack(data)
    mean_trajectory = data_array.mean(axis=0)
    
    sns.lineplot(x=np.arange(len(mean_trajectory)), y=mean_trajectory, ax=ax2,
        color=colors[g], label=rf"$r$ = {recombination_rate}", linewidth = 3)

for g, recombination_rate in enumerate(mendelian_data.keys()):
    data = mendelian_data[recombination_rate]
    for replicate in data:
        sns.lineplot(replicate, ax=ax2, color = colors[g], alpha=0.25)

ax2.set_ylabel(r'$\sum$ Neutral Displacement ($D(T)$)', fontsize=12)
ax2.set_xlabel(r'Generations ($T$)', fontsize=12)
ax2.legend(title=r"Recomb. rate ($r$)")

# ax3: rate of system drift
for g, recombination_rate in enumerate(mendelian_data.keys()):
    data = mendelian_data[recombination_rate]
    data_array = np.vstack(data)
    # derivative along time axis
    derivative = np.diff(data_array, axis=1)
    # mean derivative across replicates
    mean_derivative = derivative.mean(axis=0)
    sns.lineplot(x=np.arange(len(mean_derivative)), y=mean_derivative, ax=ax3,
        color=colors[g], label=rf"$r$ = {recombination_rate}", linewidth=2)

ax3.set_ylabel(r'Rate of Dev. Sys. Drift ($|\Delta x_t|$)', fontsize=12)
ax3.set_xlabel(r'Generations ($T$)', fontsize=12)
ax3.legend(title=r"Recomb. rate ($r$)")

# add panel labels
ax1.text(-0.18, 1.07, 'A', transform=ax1.transAxes, fontsize=14, va='top', ha='left')
ax2.text(-0.155, 1.07, 'B', transform=ax2.transAxes, fontsize=14, va='top', ha='left')
ax3.text(-0.245, 1.07, 'C', transform=ax3.transAxes, fontsize=14, va='top', ha='left')

plt.tight_layout()
plt.savefig('../figures/recombination_rate_simulations.png', dpi=600)
