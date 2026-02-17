#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D

# load data
data = pd.read_csv('../data/neutral_manifold_volume_mc_estimated.csv')
# separate data
k2 = data.loc[data['k'] == 2]
k3 = data.loc[data['k'] == 3]
k4 = data.loc[data['k'] == 4]
k5 = data.loc[data['k'] == 5]
k6 = data.loc[data['k'] == 6]

# function to add linear model
def add_linear_fit(df, color):
	X = sm.add_constant(df['tau'])
	y = df['volume']
	model = sm.OLS(y, X).fit()
	tau_vals = np.linspace(df['tau'].min(), df['tau'].max(), 100)
	pred = model.params['const'] + model.params['tau'] * tau_vals
	ax1.plot(tau_vals, pred, linestyle='--', color=color)

# initialize plot
sns.set_style('whitegrid')
fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (10,5))

# define color palette
colors = sns.color_palette("viridis", n_colors=5)

# First subplot
#k2 plot
sns.lineplot(data = k2, x = 'tau', y = 'volume', ax=ax1, color=colors[0])
add_linear_fit(k2, colors[0])

#k3 plot
sns.lineplot(data = k3, x = 'tau', y = 'volume', ax=ax1, color=colors[1])
add_linear_fit(k3, colors[1])

#k4 plot
sns.lineplot(data = k4, x = 'tau', y = 'volume', ax=ax1, color=colors[2])
add_linear_fit(k4, colors[2])

#k5 plot
sns.lineplot(data = k5, x = 'tau', y = 'volume', ax=ax1, color=colors[3])
add_linear_fit(k5, colors[3])

#k6 plot
sns.lineplot(data = k6, x = 'tau', y = 'volume', ax=ax1, color=colors[4])
add_linear_fit(k6, colors[4])

ax1.set_ylabel(r'Relative volume ($\hat{V}_{\eta(\rho)})$', fontsize=12)
ax1.set_xlabel(r'Fitness contour ($\rho)$', fontsize=12)

# --- add legend --- #
legend_elements = [
	Line2D([0], [0], color=colors[0], lw=1.5, label='k = 2'),
	Line2D([0], [0], color=colors[1], lw=1.5, label='k = 3'),
	Line2D([0], [0], color=colors[2], lw=1.5, label='k = 4'),
	Line2D([0], [0], color=colors[3], lw=1.5, label='k = 5'),
	Line2D([0], [0], color=colors[4], lw=1.5, label='k = 6'),
	Line2D([0], [0], color='black', lw=1.5, linestyle='-', label='Monte Carlo'),
	Line2D([0], [0], color='black', lw=1.5, linestyle='--', label=r'$\hat{V}_{\eta(\rho)}=\beta_{0, k}+ \beta_{1, k} \rho + \epsilon$'),
]
ax1.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=9)
ax1.set_ylim(0,1)

# Second subplot
ks = [2,3,4,5, 6]
dfs = [k2, k3, k4, k5, k6]

# fit models and store intercepts
intercepts = []
slopes = []

for df in dfs:
	X = sm.add_constant(df['tau'])
	y = df['volume']
	model = sm.OLS(y, X).fit()
	intercepts.append(model.params['const'])
	slopes.append(model.params['tau'])

intercepts = np.array(intercepts)
slopes = np.array(slopes)

# add slopes and intercepts plot
sns.pointplot(x = ks, y = intercepts, color = 'black', ax=ax2)
ax2.set_ylabel(r'Intercept ($\beta_{0,k}$)', fontsize=12)
ax2.set_xlabel(r'Free parameters ($k$)', fontsize=12)

ax2b = ax2.twinx()
sns.pointplot(x = ks, y = slopes, color = 'gray', linestyle = '-', ax=ax2b)
ax2b.set_ylabel(r'Slope ($\beta_{1,k}$)', fontsize=12, color='gray')
ax2b.tick_params(axis='y', labelcolor='grey')
ax2b.grid(False)

# add axis labels
ax1.text(-0.155, 1.06, 'A', transform=ax1.transAxes, fontsize=14, va='top', ha='left')

ax2.text(-0.175, 1.06, 'B', transform=ax2.transAxes, fontsize=14, va='top', ha='left')

plt.tight_layout()
plt.savefig('../figures/dev_sys_drift_complexity_analysis.png', dpi=600)
