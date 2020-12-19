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
import winsound
import concurrent.futures

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
  iters=151
  div = 0
  λ = 100  # population size
  μ = 2
  migration = 16 # migration between islands
  localSearch = 1   

  # Initialization
  offspring = []


  # CLUSTER INDIVIDUALS AND SPLIT INTO ISLANDS
  
  # CLUSTER INDIVIDUALS
  '''similarity = []
  refidx = fitnesses.index(min(fitnesses))
  ref = populationMixed[refidx]
  # calculate difference between best individual and rest of the population
  for ind in populationMixed:
    similarity.append(distance(ind, ref))  
  X = np.array(similarity).reshape(-1,1)
  #print('Similarity: ',X[:,0])
  # compute number of clusters (K) 
  num_islands = 3 # int(math.ceil(math.sqrt(λ) + 0.5))
  # cluster individuals 
  kmeans = KMeans(n_clusters=num_islands, random_state=0).fit(X)
  centers = kmeans.cluster_centers_'''
  #print(f'Categorization:    {kmeans.labels_}')
  #print(f'Clusters:    {set(kmeans.labels_)}')

  # SPLIT INTO ISLANDS
  num_islands = 2
  populations = []
  diversity = [div] * num_islands
  k = [5,5,3]
  trapped = [0] * num_islands
  for isl in range(num_islands):
    # Get the index of the individuals in each cluster 
    '''clusterPop = list(np.where(kmeans.labels_ == cluster)[0])
    print(f'Individuals in CLUSTER {cluster}: {len(clusterPop)}') '''   
    populations.append(initializeKMeans(tsp, λ))
    '''if (isl+1) % 2 == 0:
      populations.append(initializeKMeans(tsp, λ))
    else:
      populations.append(initialize(tsp, λ))'''
  # empty initial population
  populationMixed = []
  # PLOT CLUSTERS

  '''plt.scatter(X[:,0], np.array([1]*λ), c=kmeans.labels_, s=50, cmap='viridis')
  centroids = list(range(num_islands))
  plt.show()'''
  # apply local search to the best individuals of the population 
  
  # LOCAL SEARCH OVER FIRST INDIVIDUALS
  bestFit = []
  for population in populations:
    fitnesses = [fitness(tsp, ind.order) for ind in population]    
    bestFit.append(min(fitnesses))
    bestInd = fitnesses.index(min(fitnesses))
    population[bestInd].order = local_search(tsp, population[bestInd].order)
  
  # EVOLUTION
  for iter in range(iters):
    end_t = time.perf_counter()
    if end_t - init_t > 60 * 5:
      print('Time out')
      print('Iteration: ',iter)      
      break
    if iter < 100:
      probDiv = probIni * np.power(0.9, iter/10)
    else: probDiv = probIni * np.power(0.6, iter/10)
    populationMixed = []
    island = 0
    for population in populations:      
      offspring = []

      # Check if trapped in local

      if trapped[island] >= 3  and diversity[island] < div:
        #print('Trapped in island: ', island)
        # Increase diversity                   
        # Apply different local search to escape 
        for ind in population:          
          if island == 0 and np.random.rand() < 0.9:
              # Inverse mutation over 20% population              
              #mutate(ind, force=True)
              ind.order = np.random.permutation(ind.order)              
          elif island == 1 and np.random.rand() < 0.9:
              # Scramble mutation over 20% population              
              insertMutate(ind, force=True)
              #ind.order = np.random.permutation(ind.order)   
          elif island == 2 and np.random.rand() < 0.9:
              # Scramble mutation over 20% population              
              insertMutate(ind, force=True)  
        # Force migration
        if False: # trapped[island] >= 5
          print('URGENT!! ISLAND: ', island)                 
          # select random island
          islands = np.append(np.arange(0,island), np.arange(island + 1,num_islands), 0)
          isl = random.sample(list(islands), 1)[0] 
          # select random individuals 
          inm = random.sample(populations[isl], 1)[0]        
          population[random.randrange(len(population))] = inm
          print(f'Migrated indivudal from {isl} with fitnesses {fitness(tsp, inm.order)}')
        # add individuals to current island
      
      # Recombination
      for ind in range(μ):
        # Selection
        parent1 = selection(tsp, population, k[island])
        parent2 = selection(tsp, population, k[island])
        offspring.append(recombination(parent1, parent2, tsp))            
        if diversity[island] < div and iter < iters - 1:
          if (island == 0 or underestimate) and np.random.rand() < probDiv:
            insertMutate(offspring[ind], force=True)
          elif island == 1 and np.random.rand() < probDiv:
            insertMutate(offspring[ind], force=True)
          else:
            if np.random.rand() < probDiv:
              scrambleMutate(offspring[ind], force=True)
        else:
          if island == 0 or underestimate:
            insertMutate(offspring[ind])
          elif island == 1:
            insertMutate(offspring[ind])         
          else:
            scrambleMutate(offspring[ind])
        
        # Local search over offspring
        if (iter % 5 < 0 and iter > 0) and np.random.rand() < 0.5:
          offspring[ind].order = local_search(tsp, offspring[ind].order, underestimate=True)  


      # Mutation
      for ind in population:
        if diversity[island] < div and iter < iters - 1:
          if (island == 0 or underestimate) and np.random.rand() < probDiv:
            insertMutate(ind, force=True)
          elif island == 1 and np.random.rand() < probDiv:
            insertMutate(ind, force=True)
          else:
            if np.random.rand() < probDiv:
              scrambleMutate(ind, force=True)
        else:
          if island == 0 or underestimate:
            insertMutate(ind)
          elif island == 1:
            insertMutate(ind)         
          else:
            scrambleMutate(ind)

      # MIGRATION

      if iter % migration == 0 and iter != 0 or iter == iters -1:
        # select random island
        islands = np.append(np.arange(0,island), np.arange(island + 1,num_islands), 0)
        isl = random.sample(list(islands), 1)[0]
        # select random individuals 
        inm1 = selection(tsp, populations[isl], k[island])
        inm2 = selection(tsp, populations[isl], k[island])
        # add individuals to current island
        offspring.extend([inm1, inm2])        

      # Elimination
      population = elimination(population, offspring, tsp)      
      fitnesses = [fitness(tsp, ind.order) for ind in population]
      # Check best fitness variation
      if (bestFit[island] == min(fitnesses)):
        trapped[island] += 1    
      else:        
          trapped[island] = 0
      bestFit[island] = min(fitnesses)
      
      if iter % localSearch == 0 and iter != 0:
        bestInd = fitnesses.index(bestFit[island])
        population[bestInd].order = local_search(tsp, population[bestInd].order, underestimate=underestimate)

      # Update new population
      populations[island] = population

      # Add island population to mixed population
      if iter == iters -1:
        populationMixed.extend(population)


      #means.append(statistics.mean(fitnesses))
      #bests.append(min(fitnesses))
      diversity[island] = statistics.mean(fitnesses) - bestFit[island]
      if iter % 1 == 0:
        print(f'---------------------------------  ITERATION {iter}, ISLAND {island} ---------------------------------')  
        print("Mean fitness:  ", f'{statistics.mean(fitnesses):.5f}', '\t', "Best fitness:  ", f'{bestFit[island]:.5f}',
         "\t", 'Diversity:', f'{diversity[island]:.5f}')
      island += 1


  fitnesses = [fitness(tsp, ind.order) for ind in populationMixed]
  # Print Best final Candidate solution
  bestSol = fitnesses.index(min(fitnesses))
  print(f'\nBest solution {min(fitnesses)} mutation rate', populationMixed[bestSol].α)
  # check if it has inf
  '''for ind in population:
    print("Candidate solution:  ", ind.order)
    print("Candidate mutation probability:  ", ind.α)'''
  #generations = np.arange(iters)  
  '''plt.plot(generations, means)
  plt.plot(generations, bests)
  plt.xlabel('Generations')
  plt.ylabel('Fitness')
  plt.legend(['Mean', 'Best fitness'])
  plt.show()'''

def distance(ind, ref):
  diff = 0  
  for i in range(len(ref.order)):
    if ind.order[i] != ref.order[i]:
      diff += 1
  return math.ceil(diff/2)

def initialize(tsp, λ):
  population = []
  for i in range(λ):
    population.append(Individual(tsp))
  return population

def initialize_local_search(tsp, λ):
  population = []
  for i in range(λ):
    ind = Individual(tsp)
    ind.order = local_search_fast(tsp, ind.order)
    population.append(ind)
  return population

def local_search(tsp, order, underestimate=False):
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

def initializeKMeans(tsp, λ): 
  population = []  
  for ii in range(3):
    population.append(create_seed_ind(tsp))
  #seedOrder = list(seed.order)
  
  '''for ii in range(1,λ): 
    ind = Individual(tsp)
    ind.order = np.array(mutate_order(seedOrder[:]))
    ind.order = np.array(mutate_order(ind.order[:]))
    population.insert(ii,ind)'''
  # Random initialize the other half
  for ii in range(3,λ):
    ind = Individual(tsp)
    population.insert(ii,ind)
  return population

def mutate_order(order):
  i = random.randrange(len(order))
  j = random.randrange(len(order))
  #order[i], order[j] = order[j], order[i]  
  if i < j:
    order[i:j] = order[i:j][::-1]
  else:
    order[j:i] = order[j:i][::-1]
  return order

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

  # PLOT CLUSTERS
  '''
  plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, s=50, cmap='viridis')
  centers = kmeans.cluster_centers_
  plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
  labels = list(range(n_cities))
  centroids = list(range(K))
  #for label, x, y in zip(labels, mds_coords[:,0], mds_coords[:,1]):
      #plt.annotate(label, (x,y), xycoords = 'data')
  for centroid, x, y in zip(centroids, centers[:,0], centers[:,1]):
      plt.annotate(centroid, (x,y), xycoords = 'data', c='red')
  plt.show()'''

  return firstInd
  
def mutate(individual, force=False):
  # inversion mutation
  if np.random.rand() < individual.α or force:
    i = random.randrange(len(individual.order))
    j = random.randrange(len(individual.order))
    if i < j:
      individual.order[i:j] = individual.order[i:j][::-1]
    else:
      individual.order[j:i] = individual.order[j:i][::-1]

def scrambleMutate(individual, force=False):
  # scramble mutation
  if np.random.rand() < individual.α or force:
    i = random.randrange(len(individual.order))
    j = random.randrange(len(individual.order))
    if i < j:
      individual.order[i:j] = np.random.permutation(individual.order[i:j])
    else:
      individual.order[j:i] = np.random.permutation(individual.order[j:i])

def swapMutate(individual, force=False):
  # swap mutation
  if random.random() < individual.α or force:
    x = random.randrange(len(individual.order))
    y = random.randrange(len(individual.order))
    individual.order[x], individual.order[y] = individual.order[y], individual.order[x]

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

def recombination(parent1, parent2, tsp):
  child = Individual(tsp)
  # Select crossover points randomly
  i = random.randrange(len(parent1.order))
  j = random.randrange(len(parent1.order))
  # Perform order crossover recombination to create the child's order
  if i < j:  
    child.order = order_crossover(parent1.order, parent2.order, i, j)
  else:  
    child.order = order_crossover(parent1.order, parent2.order, j, i)    
  # Set child parameters
  if parent1.α == parent2.α:
    child.α = parent1.α
  elif np.random.rand() < 0.1:
    child.α = random.sample(list(np.linspace(0.5/len(tsp.vertices), 2.5/len(tsp.vertices), 20)), 1)[0]
  else:
    child.α = random.sample([parent1.α, parent1.α], 1)[0]  
  #print(f'REAL CH: {child.order} F: {fitness(tsp, child)}')
  #print(f'P1: {parent1.order} F: {fitness(tsp, parent1)}\nP2: {parent2.order} F: {fitness(tsp, parent2)}\nCH: {child.order} F: {fitness(tsp, child)}')
  return child
  '''β = 2 * np.random.rand() - 0.5 # coefficient between (-0.5, 1.5)
  child.α = parent1.α + β * (parent2.α - parent1.α)'''
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
  if None in child:
    print('Order crossover longitud ', len(parent1), ' == ', len(parent2))
    print('Order crossover de ', i, ' a ', j)
    print(parent1,' - ',parent2, ' vs ', child)
  return child

def selection(tsp, population, k=5):
  if len(population) < k:
    k=len(population)
  candidates = random.sample(population, k)
  fitnesses = [fitness(tsp, ind.order) for ind in candidates]
  selected = fitnesses.index(min(fitnesses))
  return candidates[selected]

def elimination(population, offspring, tsp):
  # λ+μ elimination
  combined = population + offspring
  ranked_individuals = sorted([(fitness(tsp, ind.order), ind) for ind in combined], key=lambda ind: ind[0])[0:len(population)]
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