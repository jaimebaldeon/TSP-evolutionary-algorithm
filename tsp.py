# -*- coding: utf-8 -*-
import numpy as np
import statistics
import random
import sys
import matplotlib.pyplot as plt
import time
from sklearn import manifold
import math
from sklearn.cluster import KMeans

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
    self.α = random.sample(list(np.linspace(0.5/len(tsp.vertices), 2.5/len(tsp.vertices), 20)), 1)[0]

def fitness(tsp, order):
  value = 0
  ii = 0
  for i in order:
    # Sum up total distance for the individual's sequence of nodes 
    while (ii < order.size - 1):
      pairVertices = order[ii:ii+2]
      value += tsp.distanceMatrix[pairVertices[0], pairVertices[1]]
      ii += 1 
  # add the distance between first and last node
  value += tsp.distanceMatrix[order[-1], order[0]]
  return value

def optimize(tsp):
  init_t = time.perf_counter()
  λ = 20 # population size
  μ = 180 # offspring size
  iters =9000
  migration = 50 # migration between islands
  localSearch = 2

# INITIALIZATION

  # SPLIT INTO ISLANDS

  num_islands = 2
  populations = []
  populationMixed = []
  for isl in range(num_islands):     
    populations.append(initializeKMeans(tsp, λ))    

  diversity = [0] * num_islands
  trapped = [0] * num_islands
  means = [0] * num_islands
  bests = [0] * num_islands

# EVOLUTION

  for iter in range(iters):  	
    end_t = time.perf_counter()
    if end_t - init_t > 60 * 5:
      print('Time out')
      print('Iteration: ',iter)
      break 
  
  # FOR EACH ISLAND

    island = 0
    for population in populations:
    	offspring = []    

    # RECOMBINATION

    	for ind in range(μ):
    		# Selection
    		parent1 = selection(tsp, population)
    		parent2 = selection(tsp, population)
    		if island % 3 == 0: 
    			child1, child2 = pmx_recombination(parent1, parent2, tsp)
    			inverseMutate(child1)
    			inverseMutate(child2)
    			offspring.extend([child1, child2])
    		elif island % 3 == 1:
    			offspring.append(recombination(parent1, parent2, tsp))
    			assortedMutate(offspring[ind], iter)
    		else: 
    			offspring.append(recombination(parent1, parent2, tsp))
    			insertMutate(offspring[ind])    		    		    		    		

    	# LOCAL SEARCH OVER OFFSPRING

    		#offspring[ind].order = local_search_fast(tsp, offspring[ind].order)  


    # MUTATION

    	for ind in population:
    		if island % 3 == 0:	inverseMutate(ind)
    		elif island % 3 == 1: assortedMutate(ind, iter)
    		else: insertMutate(ind)

    # MIGRATION

    	if iter % migration == 0 and iter != 0:
    		# select random island
    		islands = np.append(np.arange(0,island), np.arange(island + 1,num_islands), 0)
    		isl = random.sample(list(islands), 1)[0]
    		# select random individuals 
    		inm1 = selection(tsp, populations[isl])
    		inm2 = selection(tsp, populations[isl])

    		# add individuals to current island
    		offspring.extend([inm1, inm2])

    # ELIMINATION

    	population = elimination(population, offspring, tsp)    	    

    # LOCAL SEARCH OVER BEST SOLUTIONS

    	fitnesses = [fitness(tsp, ind.order) for ind in population]
    	if iter % localSearch == 0 and iter != 0 and trapped[island] < 5:
    	    	    	    		bestInd = fitnesses.index(min(fitnesses))
    	    	    	    		population[bestInd].order = local_search(tsp, population[bestInd].order)
    	
    # CHECK IF TRAPPED IN LOCAL SPACE  

    	if trapped[island] < 5: fitnesses = [fitness(tsp, ind.order) for ind in population]
    	if (bests[island] == min(fitnesses)):
    		trapped[island] += 1
    	else:        
    		trapped[island] = 0       

	# UPDATE NEW POPULATION

    	bests[island] = min(fitnesses)
    	populations[island] = population	      

    # MIX ISLANDS IN THE LAST ITERATION 

    	if iter == iters -1:
    		populationMixed.extend(population)

    # HISTORICAL DATA GATHERING    	
    			
    	means[island] = statistics.mean(fitnesses)    	
    	diversity[island] = statistics.mean(fitnesses) - bests[island]
    	if iter % 1 == 0:
    		print(f'---------------------------------  ITERATION {iter}, ISLAND {island} ---------------------------------')  
    		print("Mean fitness:  ", f'{statistics.mean(fitnesses):.5f}', '\t', "Best fitness:  ", f'{bests[island]:.5f}',
         	"\t", 'Diversity:', f'{diversity[island]:.5f}')
    	island += 1

# BEST SOLUTION 

  # Print Best final Candidate solution
  fitnesses = [fitness(tsp, ind.order) for ind in populationMixed]
  # Print Best final Candidate solution
  bestSol = fitnesses.index(min(fitnesses))
  print(f'\nBest solution {min(fitnesses)} mutation rate', populationMixed[bestSol].α)

  '''generations = np.arange(iters)
  for island in range(num_islands):
  	plt.plot(generations, means[island][1])
  	plt.plot(generations, bests[island][1])
  	plt.xlabel('Generations')
  	plt.ylabel('Fitness')
  	plt.legend(['Mean', 'Best fitness'])
  	plt.show()'''

def initialize(tsp, λ):
  population = []
  for i in range(λ):
    population.append(Individual(tsp))
  return population

def initializeKMeans(tsp, λ): 
  population = []  
  for ii in range(3):
    population.append(create_seed_ind(tsp))
    
  # Random initialize the other half
  for ii in range(3,λ):
    ind = Individual(tsp)
    population.insert(ii,ind)
  return population

def create_seed_ind(tsp):

  # TRANSFORM MATRIX INTO 2D SPACE

  matrix = tsp.distanceMatrix
  inf = float('inf')
  if np.max(matrix) == inf:
    # infinite weights must be changed to finite values
    matrix = np.where(matrix==inf,1e8, matrix)
  sym_matrix = np.maximum( matrix, matrix.transpose() )
  mds_model = manifold.MDS(n_components = 2, random_state = 123,
      dissimilarity = 'precomputed')
  mds_fit = mds_model.fit(sym_matrix)  
  mds_coords = mds_model.fit_transform(sym_matrix) 
  # Create list with cities coordinates
  coords = []
  for i in range(len(mds_coords[:,0])):
    coords.append(list((mds_coords[i,0], mds_coords[i,1])))
  # transform coordinates to np array
  X = np.array(coords) 


  # CLUSTER CITIES IN K GROUPS

  # compute number of clusters (K) 
  n_cities = len(tsp.vertices)
  K = int(math.ceil(math.sqrt(n_cities) + 0.5))
  # cluster cities 
  kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
  centers = kmeans.cluster_centers_
  centersDict = dict(zip(list(range(len(centers))), centers))

  # COMPUTE SOLUTION FOR EACH CLUSTER AND JOIN THEM TO GET FINAL TOUR SOLUTION

  firstSol = [] # final solution list
  cluster = 0
  for i in range(K):
    clusterSol = [] # cluster solution
    # select random city from each cluster and add to solution tour
    clusterInst = np.where(kmeans.labels_ == cluster)[0]    
    startInst = random.choice(clusterInst)
    clusterSol.append(startInst)
    # delete selected city from cluster 
    clusterInst = np.delete(clusterInst, np.where(clusterInst == startInst))
    nclusters = len(clusterInst)
    # add the nearest city as new starting city until having all cities connected
    for city in range(nclusters):     
      # get distance to the other cities
      dist = list(matrix[startInst][clusterInst])
      # Get nearest city
      nearest = clusterInst[dist.index(min(dist))]
      # append closest city
      clusterSol.append(nearest)
      # update starting city
      startInst = nearest
      # delete selected city from cluster 
      clusterInst = np.delete(clusterInst, np.where(clusterInst == startInst))

    # Find closest cluster   

    actual = centers[cluster]
    #print(f'Actual cluster {cluster} with position {actual}')
    clustDist = []
    for center in centersDict.values():
      # compute distances to non-visited clusters
      if np.linalg.norm(actual-center) > 0:
        clustDist.append(np.linalg.norm(actual-center)) 
      else:
        # same cluster cannot be selected
        clustDist.append(float('inf'))  

    # Get closest cluster and update to be computed next

    previous = cluster
    idx = clustDist.index(min(clustDist))   
    cluster = list(centersDict)[idx]
    #print(f'Cluster actual: {previous}. Closest cluster: {cluster}')     
    # Remove actual cluster from dict   
    centersDict.pop(previous)

    # Append tour solution of the computed cluster

    firstSol.extend(clusterSol)       

  # CREATE FIRST INDIVIDUAL OF POPULATION
  firstInd = Individual(tsp)  
  firstInd.order = np.array(firstSol)

  return firstInd

def local_search(tsp, order):
  fitnesses = []
  neighbours = []
  fitnesses.append(fitness(tsp, order))
  neighbours.append(order)
  n = 1
  while(n <= len(order) - 3):
    for i in range(len(order)):
      cycleSeq = np.append(order, order)
      e1 = cycleSeq[i-1]
      e2 = cycleSeq[i+n+1]
      if i == 0:
        idx = np.where(cycleSeq == e1)[0][0]
      else:
        idx = np.where(cycleSeq == e1)[0][1]
      inv = cycleSeq[i+n+1:idx+1][::-1]
      rem = cycleSeq[i:i+n+1]
      newOrder = np.append(rem, inv)
      neighbours.append(newOrder)
      # calculate fitness of order and append to list
      fitnesses.append(fitness(tsp, newOrder))
    n+=1
  selected = fitnesses.index(min(fitnesses))
  return  neighbours[selected]

def local_search_fast(tsp, order):
  fitnesses = []
  neighbours = []
  fitnesses.append(fitness(tsp, order))
  neighbours.append(order)
  maxN = len(order) - 3
  n = 1
  stopMiddle = False
  while(n <= maxN):
    if len(order) % 2 == 0:
      # even
      if n == math.ceil(maxN/2):
        stopMiddle = True
    else:
      # uneven
      if n > maxN/2:
        break
    for i in range(len(order)):
      if stopMiddle and i == len(order)/2:
        n = maxN      
        break
      cycleSeq = np.append(order, order)
      e1 = cycleSeq[i-1]
      e2 = cycleSeq[i+n+1]
      if i == 0:
        idx = np.where(cycleSeq == e1)[0][0]
      else:
        idx = np.where(cycleSeq == e1)[0][1]
      inv = cycleSeq[i+n+1:idx+1][::-1]
      rem = cycleSeq[i:i+n+1]
      newOrder = np.append(rem, inv)
      neighbours.append(newOrder)
      # calculate fitness of order and append to list
      fitnesses.append(fitness(tsp, newOrder))
    n+=1
  selected = fitnesses.index(min(fitnesses))
  return  neighbours[selected]

def inverseMutate(individual):
  # inverse mutation
  if np.random.rand() < individual.α:
    i,j = 0,0
    while(i == j):
    	i = random.randrange(len(individual.order))
    	j = random.randrange(len(individual.order))
    if i < j:
      individual.order[i:j] = individual.order[i:j][::-1]
    else:
      individual.order[j:i] = individual.order[j:i][::-1]

def scrambleMutate(individual, force=False):
  # scramble mutation
  if np.random.rand() < individual.α or force:
    i,j = 0,0
    while(i == j):
    	i = random.randrange(len(individual.order))
    	j = random.randrange(len(individual.order))
    if i < j:
      individual.order[i:j] = np.random.permutation(individual.order[i:j])
    else:
      individual.order[j:i] = np.random.permutation(individual.order[j:i])

def swapMutate(individual, force=False):
  # swap mutation
  if random.random() < individual.α or force:
    i,j = 0,0
    while(i == j):
    	i = random.randrange(len(individual.order))
    	j = random.randrange(len(individual.order))
    individual.order[i], individual.order[j] = individual.order[j], individual.order[i]

def insertMutate(individual, force=False):
  # insert mutation
  if random.random() < individual.α or force:
    i,j = 0,0
    while(i == j):
    	i = random.randrange(len(individual.order))
    	j = random.randrange(len(individual.order))
    aux = individual.order
    if i < j:
      individual.order = np.concatenate(
      	(individual.order[:i],individual.order[i],individual.order[j],individual.order[i+1:j],individual.order[j+1:]), axis=None)
    else:
      individual.order = np.concatenate(
      	(individual.order[:j],individual.order[j],individual.order[i],individual.order[j+1:i],individual.order[i+1:]), axis=None)

def assortedMutate(individual, iter):
	if iter % 4 == 0:
		swapMutate(individual)
	elif iter % 4 == 1:
		insertMutate(individual)
	elif iter % 4 == 2:
		scrambleMutate(individual)
	elif iter % 4 == 3:
		inverseMutate(individual)

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
  if parent1.α == parent2.α:
  	child.α = parent1.α
  elif np.random.rand() < 0.1:
  	child.α = random.sample(list(np.linspace(0.5/len(tsp.vertices), 2.5/len(tsp.vertices), 20)), 1)[0]
  else:
  	child.α = random.sample([parent1.α, parent2.α], 1)[0]  
  #print(f'REAL CH: {child.order} F: {fitness(tsp, child)}')
  #print(f'P1: {parent1.order} F: {fitness(tsp, parent1)}\nP2: {parent2.order} F: {fitness(tsp, parent2)}\nCH: {child.order} F: {fitness(tsp, child)}')
  return child

def pmx(a,b, start, stop):
	child = np.array([None]*len(a))
	# Copy a slice from first parent:
	child[start:stop] = a[start:stop]
	# Map the same slice in parent b to child using indices from parent a:
	for ind,x in enumerate(b[start:stop]):
		ind += start
		if x not in child:
			while child[ind] != None:
				ind = np.where(b == a[ind])[0]
			child[ind] = x
	# Copy over the rest from parent b
	for ind,x in enumerate(child):
		if x == None:
			child[ind] = b[ind]
	return child

def pmx_recombination(parent1, parent2, tsp):
	child1 = Individual(tsp)
	child2 = Individual(tsp)
	half = len(parent1.order) // 2
	start = random.randint(0, len(parent1.order)-half)
	stop = start + half
	child1.order, child2.order = pmx(parent1.order,parent2.order,start,stop) , pmx(parent2.order,parent1.order,start,stop)
	# Set child parameters  
	if parent1.α == parent2.α:
		child1.α, child2.α = parent1.α, parent1.α
	elif np.random.rand() < 0.1:
		child1.α = random.sample(list(np.linspace(0.5/len(tsp.vertices), 2.5/len(tsp.vertices), 20)), 1)[0]
		child2.α = random.sample(list(np.linspace(0.5/len(tsp.vertices), 2.5/len(tsp.vertices), 20)), 1)[0]
	else:
		child1.α = random.sample([parent1.α, parent2.α], 1)[0]
		child2.α = random.sample([parent1.α, parent2.α], 1)[0]
	return child1, child2

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
  k = 2
  candidates = random.sample(population, k)
  fitnesses = [fitness(tsp, ind.order) for ind in candidates]
  selected = fitnesses.index(min(fitnesses))
  return candidates[selected]

def elimination(population, offspring, tsp):
  # λ+μ elimination
  combined = population + offspring
  ranked_individuals = sorted([(fitness(tsp, ind.order), ind) for ind in combined], key=lambda ind: ind[0])[0:len(population)]
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