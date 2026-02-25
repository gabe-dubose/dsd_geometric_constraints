#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator


# load fitness landscape
fitness_landscape = pd.read_csv('../data/fitness_landscape_for_simulations_large.csv', index_col = 0)

# convert index/columns to numeric
a_values = fitness_landscape.index.astype(float).values
b_values = fitness_landscape.columns.astype(float).values
Z = fitness_landscape.values.astype(float)

# interpolate fitness landscape
# build interpolator
fitness_interpolator = RegularGridInterpolator((a_values, b_values), Z, 
	bounds_error=False, fill_value=0.0)
	
# estimate linear function of high fitness ridge through a,b
A_grid, B_grid = np.meshgrid(a_values, b_values, indexing = 'ij')
coords = np.column_stack([A_grid.ravel(), B_grid.ravel()])
fitness_flat = Z.ravel()
threshold = np.percentile(fitness_flat, 95)
ridge_points = coords[fitness_flat >= threshold]
a_ridge = ridge_points[:, 0]
b_ridge = ridge_points[:, 1]
# fit line b = m*a + c
m,c = np.polyfit(a_ridge, b_ridge, 1)
# define ridge direction
ridge_direction = np.array([1, m])
ridge_direction = ridge_direction / np.linalg.norm(ridge_direction)

# function to get interpolated fitness from fitness landscape (vectorized)
def calcualte_population_fitness(population):
    return fitness_interpolator(population)

# function to mutate a,b vector
def mutate(ab, sigma=0.1):
	mutation = np.random.normal(loc=0, scale=sigma, size=2)
	ab_mutated = ab + mutation
	return(ab_mutated)

# function to conduct recombination
def recombine(parent1, parent2, genetic_architecture, segregation_sigma=0.02):
	
	# mendelian trait
	if genetic_architecture == 'mendelian':
		# get a from one parent and b from the other
		if np.random.rand() < 0.5:
			return np.array([parent1[0], parent2[1]])
		else:
			return np.array([parent2[0], parent1[1]])
	
	# quantitative trait
	if genetic_architecture == 'quantitative':
		# use midparent + segregation variance
		midparent = (parent1 + parent2) / 2
		segregation = np.random.normal(0, segregation_sigma)
		return midparent + segregation

def evolve(init_a, init_b, genetic_architecture, recombination_rate, n, generations, verbose = False):
	
	init_ab = np.array([init_a, init_b])
	population = np.array([mutate(init_ab) for _ in range(n)])
	
	# vectors and scalars for tracking dynamics
	mean_history = []
	step_sizes = []
	variance_history = []
	cumulative_displacement = 0
	
	# compute initial
	mean_history.append(np.mean(population, axis=0))
	variance_history.append(np.var(population[:,0]) + np.var(population[:,1]))
	
	mu0 = mean_history[0]
	prev_x = 0.0
	
	# evolutionary loop
	for generation in range(generations):
		
		# verbose
		if verbose == True:
			if generation % 10 == 0:
				print(f"Working on generation: {generation}")
		
		# --- Selection --- #
		fitness = calcualte_population_fitness(population)
		fitness = np.maximum(fitness, 0)
		if np.sum(fitness) == 0:
			probs = np.ones(n) / n
		else:
			probs = fitness / np.sum(fitness)
		parent_indices = np.random.choice(np.arange(n), size=n, p=probs)
		selected_parents = population[parent_indices]
		
		next_generation = []
		# --- Recombination --- #
		for i in range(n):
			# recombination reproduction
			if np.random.rand() < recombination_rate:
				# pick parents at randome from selected pool
				p1, p2 = selected_parents[np.random.choice(n, size=2, replace=False)]
				offspring = recombine(p1, p2, genetic_architecture)
			else:
				# clonal reproduction
				offspring = selected_parents[i]
			
			# --- Mutation --- #
			offspring = mutate(offspring)
			next_generation.append(offspring)
			
		population = np.array(next_generation)
		
		# population tracking
		
		# mean
		current_mean = np.mean(population, axis=0)
		mean_history.append(current_mean)
		
		# drift along ridge
		x_t = np.dot(current_mean - mu0, ridge_direction)
		delta_x = x_t - prev_x
		step_sizes.append(abs(delta_x))
		cumulative_displacement += abs(delta_x)
		prev_x = x_t
		
		# variance
		total_variance = np.var(population[:,0]) + np.var(population[:,1])
		variance_history.append(total_variance)
		
	
	# net displacement
	net_displacement = abs(prev_x)
	
	return {
        "mean_history": list(mean_history),
        "step_sizes": list(step_sizes),
        "cumulative_displacement": int(cumulative_displacement),
        "net_displacement": int(net_displacement),
        "variance_history": list(variance_history)
    }

recombination_rates = [0, 0.25, 0.5, 0.75, 1]
replicate_simulations = 15

# initialize dictionaries to store results
quantitative_architecture_dynamics = {}
quantitative_architecture_population_mean_history = {}

# run simulations across recombination rates for quantitative architecture
for recombination_rate in recombination_rates:
	for i in range(replicate_simulations):
		print(f"Quant: Working on replicate {i} for recombination rate {recombination_rate}")
		simulation = evolve(init_a = 6, init_b = 5, genetic_architecture = 'quantitative', 
			recombination_rate = recombination_rate, n=1000, generations=250, verbose = False)
			
		# compute cumulative displacement at each time
		cumulative_displacement_history = [float(value) for value in np.cumsum(simulation['step_sizes'])]
		# add to dynamics
		if recombination_rate not in quantitative_architecture_dynamics:
			quantitative_architecture_dynamics[recombination_rate] = []
			quantitative_architecture_dynamics[recombination_rate].append(cumulative_displacement_history)
		else:
			quantitative_architecture_dynamics[recombination_rate].append(cumulative_displacement_history)

	# save representative mean history
	quantitative_architecture_population_mean_history[recombination_rate] = [
    [float(a), float(b)] for a, b in simulation['mean_history']]

# save representative mean histories to file
file = '../data/quantitative_architecture_mean_trait_histories.json'
with open(file, 'w') as outfile:
	json.dump(quantitative_architecture_population_mean_history, outfile, indent=4)

# save to file
file = '../data/quantitative_architecture_dsd_dynamics.json'
with open(file, 'w') as outfile:
	json.dump(quantitative_architecture_dynamics, outfile, indent=4)


# initialize dictionary to store results
mendelian_architecture_dynamics = {}
mendelian_architecture_population_mean_history = {}

# run simulations across recombination rates for mendelian architecture
for recombination_rate in recombination_rates:
	for i in range(replicate_simulations):
		print(f"Mendel: Working on replicate {i} for recombination rate {recombination_rate}")
		simulation = evolve(init_a = 6, init_b = 5, genetic_architecture = 'mendelian', 
			recombination_rate = recombination_rate, n=1000, generations=250, verbose = False)
			
		# compute cumulative displacement at each time
		cumulative_displacement_history = [float(value) for value in np.cumsum(simulation['step_sizes'])]
		# add to dynamics
		if recombination_rate not in mendelian_architecture_dynamics:
			mendelian_architecture_dynamics[recombination_rate] = []
			mendelian_architecture_dynamics[recombination_rate].append(cumulative_displacement_history)
		else:
			mendelian_architecture_dynamics[recombination_rate].append(cumulative_displacement_history)
			
	# save representative mean history
	mendelian_architecture_population_mean_history[recombination_rate] = [
    [float(a), float(b)] for a, b in simulation['mean_history']]

# save representative mean histories to file
file = '../data/mendelian_architecture_mean_trait_histories.json'
with open(file, 'w') as outfile:
	json.dump(mendelian_architecture_population_mean_history, outfile, indent=4)

# save to file
file = '../data/mendelian_architecture_dsd_dynamics.json'
with open(file, 'w') as outfile:
	json.dump(mendelian_architecture_dynamics, outfile, indent=4)
