# -*- coding: utf-8 -*-
import numpy as np
import statistics
import random
import sys
import matplotlib.pyplot as plt
import time

class TSP:
  def __init__(self, matrix):
    #self.vertices = [n for n in range(1, numVertices+1)] #to accomplish natural values for vertices
    self.vertices = [n for n in range(matrix[0].size)]  # vertices from 0 to N
    inf = float('inf')
    if np.max(matrix) == inf:
      # infinite weights must be changed to finite values
      self.distanceMatrix = np.where(matrix==inf,1e8, matrix)
    else:
      self.distanceMatrix = matrix

class Individual:
  def __init__(self, tsp):
    self.order = np.random.permutation(tsp.vertices)
    self.α = random.uniform(5e-4,1e-1)

def fitness(tsp, ind):
  value = 0
  ii = 0
  for i in ind.order:
    # Sum up total distance for the individual's sequence of nodes 
    while (ii < ind.order.size - 1):
      pairVertices = ind.order[ii:ii+2]
      value += tsp.distanceMatrix[pairVertices[0], pairVertices[1]]
      ii += 1 
  # add the distance between first and last node
  value += tsp.distanceMatrix[ind.order[-1], ind.order[0]]
  return value

def optimize(tsp):
  λ = 100 # population size
  μ = 500 # offspring size
  iter =50

  # Initialization
  population = initialize(tsp,λ)
  offspring = []
  print(f"pop = {λ}, offspring = {μ}, alpha = {population[0].α}")
  means = []
  bests = []
  # Print initial Candidate solutions
  '''for ind in population:
    print("Candidate solution:  ", ind.order)'''

  # Print initial Population evaluation
  fitnesses = [fitness(tsp, ind) for ind in population]
  print("\nMean fitness:  ", statistics.mean(fitnesses))
  print("Best fitness:  ", min(fitnesses))

  for i in range(iter):
    # Recombination
    for ind in range(μ):
      # Selection
      parent1 = selection(tsp, population)
      parent2 = selection(tsp, population)
      offspring.append(recombination(parent1, parent2, tsp))
      mutate(offspring[ind])

    # Mutation
    for ind in population:
      mutate(ind)    

    # Elimination
    population = elimination(population, offspring, tsp)

    fitnesses = [fitness(tsp, ind) for ind in population]
    means.append(statistics.mean(fitnesses))
    bests.append(min(fitnesses))
    if i % 50 == 0:
      print(f'---------------------------------  ITERATION {i} ---------------------------------')   
    print("Mean fitness:  ", f'{statistics.mean(fitnesses):.5f}', '\t', "Best fitness:  ", f'{min(fitnesses):.5f}')


  # Print Best final Candidate solution
  bestSol = fitnesses.index(min(fitnesses))
  print('Best solution mutation rate', population[bestSol].α)
  # check if it has inf
  '''for ind in population:
    print("Candidate solution:  ", ind.order)
    print("Candidate mutation probability:  ", ind.α)'''
  generations = np.arange(iter)  
  plt.plot(generations, means)
  plt.plot(generations, bests)
  plt.xlabel('Generations')
  plt.ylabel('Fitness')
  plt.legend(['Mean', 'Best fitness'])
  plt.show()

def initialize(tsp, λ):
  population = []
  for i in range(λ):
    population.append(Individual(tsp))
  return population

def mutate(individual):
  # scramble mutation
  if np.random.rand() < individual.α:
    i = random.randrange(len(individual.order))
    j = random.randrange(len(individual.order))
    if i < j:
      individual.order[i:j] = individual.order[i:j][::-1]
    else:
      individual.order[j:i] = individual.order[j:i][::-1]

def recombination(parent1, parent2, tsp):
  child = Individual(tsp)
  # Select crossover points randomly
  i = random.randrange(len(parent1.order))
  j = random.randrange(len(parent1.order))
  # Perform order crossover recombination to create the child's order
  #print('ENTRO RECOMBINATION...')
  if i < j:  
    child.order = order_crossover(parent1.order, parent2.order, i, j)
    #print(f'CH: {child.order} F: {fitness(tsp, child)}')
    #print('FUNCTION: ' ,child.order)
  else:  
    child.order = order_crossover(parent1.order, parent2.order, j, i)
    #print(f'CH: {child.order} F: {fitness(tsp, child)}')
    #print('FUNCTION: ' ,child.order)
  # Set child parameters
  β = 2 * np.random.rand() - 0.5 # coefficient between (-0.5, 1.5)
  child.α = parent1.α + β * (parent2.α - parent1.α)
  #print(f'REAL CH: {child.order} F: {fitness(tsp, child)}')
  #print(f'P1: {parent1.order} F: {fitness(tsp, parent1)}\nP2: {parent2.order} F: {fitness(tsp, parent2)}\nCH: {child.order} F: {fitness(tsp, child)}')
  return child

def order_crossover(parent1, parent2, i,j):
  # Create loop effect over array
  p1 = np.concatenate((parent1, parent1), axis=None)
  p2 = np.concatenate((parent2, parent2), axis=None)
  # Create empty child order
  child = np.full(len(parent2), None) 
  # Fill child with random selected sequence from parent1
  child[i:j] = p1[i:j]
  # Set pointers to the second crossover point in parent2 (jj) and child (ii)
  ii, jj = j, j
  # The recombination stops when all nodes in parent2 sequence have been added to the child
  while (jj < j + len(parent2)):
    if ii > len(parent2)-1:
      # when child's pointer reaches the end of the array go to position 0
      ii = 0
    if p2[jj] not in child: 
      # Missing nodes are added to the child
      child[ii] = p2[jj]
      ii += 1 # when added a new node, shift child's pointer to the left
    jj += 1
  #print(f'P1: {parent1}\nP2: {parent2}\nCH: {child}')
  #print('RECOMBINATION: ' ,child)
  return child

def selection(tsp, population):
  k = 5
  candidates = random.sample(population, k)
  fitnesses = [fitness(tsp, ind) for ind in candidates]
  selected = fitnesses.index(min(fitnesses))
  return candidates[selected]

def elimination(population, offspring, tsp):
  # λ+μ elimination
  #print(sorted([(fitness(tsp, ind), ind) for ind in population], key=lambda ind: ind[0]))
  #print(sorted([(fitness(tsp, ind), ind) for ind in offspring], key=lambda ind: ind[0]))
  combined = population + offspring
  ranked_individuals = sorted([(fitness(tsp, ind), ind) for ind in combined], key=lambda ind: ind[0])[0:len(population)]
  #print(ranked_individuals)
  return [ind for fit, ind in ranked_individuals]


 

# Initialize timer
init_t = time.perf_counter()

filename = sys.argv[1]
print(f'Solving {filename}... ')
matrix = np.genfromtxt(filename,delimiter=',')
# Create TSP problem class
tsp = TSP(matrix) 

# Execution of the EA 
optimize(tsp)

# Print processing time
end_t = time.perf_counter()
print(f"Total time:     {end_t - init_t:.4f} seconds")

# 3000 + 3000 (50its)  >>  22670.967669271264
# 3000 + 3000 (100its)  >>  12837.643185583667