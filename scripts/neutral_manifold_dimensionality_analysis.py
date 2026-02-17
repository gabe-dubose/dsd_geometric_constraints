#!/usr/bin/env python3

import numpy as np
from joblib import Parallel, delayed
import itertools
import csv

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
	
# --- MONTE CARLO FUNCTIONS --- #

# function for sampling theta
def sample_theta(theta_0, free_idx, scale=0.2):
	theta = theta_0.copy()
	for idx in free_idx:
		theta[idx] = theta_0[idx] * np.random.uniform(1 - scale, 1 + scale)
	return theta

# function for Monte Carlo volume estimations
def volume_estimator(free_idx, theta_0, u_star, sigma, tau, n_mc):
	count = 0
	
	for _ in range(n_mc):
		# sample theta
		theta = sample_theta(theta_0, free_idx)
		# simulate pattern
		u, _ = rd(*theta)
		# calcualte fitness
		W = calculate_fitness(u, u_star, sigma)
		
		if W >=tau:
			count += 1

	return count / n_mc

# functionf for main simulation
def simulation():
	np.random.seed(42)
	# PARAMETERS
	# initial theta
	theta0 = [1.2, 1.0, 1.0, 1.0, 0.01, 0.1]  # a, b, c, d, Du, Dv
	# strength of stabilizing selection (sigma)
	sigma = 0.9
	# thresholds (tau)
	tau_values = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
	# free parameters (k)
	k_values = [2,3,4,5,6]
	# number of Monte Carlo simulations
	n_mc = 2000
	# maximum number of free parameter resampling
	max_subset = 50
	# optimal phenotype
	u_star, _ = rd(*theta0)
	
	#set up book keeping
	all_indices = list(range(6))
	replicate_id = 0
	results = []
	
	# run simulation
	for k in k_values:
		all_subsets = list(itertools.combinations(all_indices, k))
		
		if len(all_subsets) > max_subset:
			idx = np.random.choice(len(all_subsets), max_subset, replace=False)
			subsets = list(np.array(all_subsets, dtype=object)[idx])
		else:
			subsets = all_subsets
		
		for tau in tau_values:
			print(f"Running k={k}, tau={tau:.2f}")
			volumes = Parallel(n_jobs=-1)(
				delayed(volume_estimator)(subset, theta0, u_star, sigma, tau, n_mc) for subset in subsets)
			for V in volumes:
				results.append([replicate_id, k, tau, V])
				replicate_id += 1
		
	# save results
	outfile = '../data/neutral_manifold_volume_mc_estimated.csv'
	with open(outfile, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["replicate_id", "k", "tau", "volume"])
		writer.writerows(results)
	

if __name__ == "__main__":
	simulation()
		
