#!/usr/bin/env python3

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# --- MODEL FUNCTIONS --- #

# function to calcualte the Laplacian operator
def laplacian(f, dx):
	lap = np.zeros_like(f)
	lap[1:-1] = f[2:] - 2 * f[1:-1] + f[:-2]
	lap[0] = f[1] - 2 * f[0] + f[-1]
	lap[-1] = f[0] - 2 * f[-1] + f[-2]
	return lap / dx**2

# function to simulate reaction diffusion model
def rd(a, b, c, d, Du, Dv, L=10, dx = 0.05, dt = 0.01, T = 50, seed=None):
	# set random seed
	if seed is not None:
		np.random.seed(seed)
	
	# set space
	N = int(L / dx)
	x = np.linspace(0, L, N)
	
	# initialize u and v
	u = 0.5 + 0.01 * np.sin(2 * np.pi * x/L) + 0.01 * np.random.randn(N)
	v = 0.5 + 0.01 * np.sin(2 * np.pi * x/L) + 0.01 * np.random.randn(N)
	
	# set time
	time = int(T / dt)
	
	# run model
	for t in range(time):
		Lu = laplacian(u, dx)
		Lv = laplacian(v, dx)
		u += dt * (a * u - b * u * v + Du * Lu)
		v += dt * (c * u**2 - d * v + Dv * Lv)
	
	return u, v
	
# function to calcualte fitness
def calculate_fitness(u, u_star, sigma):
	diff = u - u_star
	dist2 = np.sum(diff**2)
	fitness = np.exp(-dist2 / (2 * sigma**2))
	return fitness
	
# function for parallel computation
def compute_point(value_i, value_j):
	theta_var = theta.copy()
	theta_var[i] = value_i
	theta_var[j] = value_j
	u, v = rd(a=theta_var[0], b=theta_var[1], c=theta_var[2], d=theta_var[3],
		Du=theta_var[4], Dv=theta_var[5], L=10, dx = 0.05, dt = 0.01, T = 50, seed=None)
	fitness = calculate_fitness(u, u_star, sigma)
	return fitness

# --- SIMULATIONS --- #

# PARAMETERS AND INIT
theta = [1.2, 1.0, 1.0, 1.0, 0.01, 0.1] # --- [a, b, c, d, Du, Dv]
sigma = 0.9 # --- strength of stabilizing selection

# initialize u_star
u_star, v_star = rd(a=theta[0], b=theta[1], c=theta[2], d=theta[3], 
	Du=theta[4], Dv=theta[5], L=10, dx = 0.05, dt = 0.01, T = 50, seed=None)

# LANDSCAPE DEFINITION
parameter = ['a', 'b', 'c', 'd', 'Du', 'Dv']
i, j = 0, 1 # --- can change to vary which parameters are used
parameter_i_range = np.linspace(0.1*theta[i], 10*theta[i], 1500)
parameter_j_range = np.linspace(0.1*theta[j], 10*theta[j], 1500)
X, Y = np.meshgrid(parameter_j_range, parameter_i_range)
Z = np.zeros_like(X)

# flatten grid for parallel runs
points = [(value_i, value_j) for value_i in parameter_i_range for value_j in parameter_j_range]

# run simulations
simulation_results = Parallel(n_jobs=-1)(
	delayed(compute_point)(val_i, val_j) for val_i, val_j in points
)

# reshape back to grid
Z = np.array(simulation_results).reshape(
	len(parameter_i_range),
	len(parameter_j_range)
)

# put together fitness landscape
a_values = list(parameter_i_range)
b_values = list(parameter_j_range)
fitness_landscape = pd.DataFrame(Z, index=a_values, columns = b_values)

# write to file
fitness_landscape.to_csv('../data/fitness_landscape_for_simulations_large.csv')
