import Reporter
import numpy as np
import random
from sklearn import manifold
import math
from sklearn.cluster import KMeans
import statistics

class r0819397:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Create TSP problem instance
		tsp = TSP(distanceMatrix)

		λ = 10 # population size
		μ = 30 # offspring size
		localSearch = 1 # perform local search every step
		probLS = 0 # Percentage of local search on offspring 

		tourLength = len(tsp.vertices) 
		if tourLength <= 65:
			num_islands = 4 # number of islands
			migration = 50 # migration between islands  	
			nbhoodLevel = 2 # number of neighborhoods to explore in local search
			λopt = 0 # Number of enhanced individuals generated with kmeans initializer
			convCounts = 35 # Number of generations to be considered as converged
		elif 65 < tourLength <= 145:
			num_islands = 4
			migration = 50 
			nbhoodLevel = 5 
			λopt = 1
			convCounts = 50
		elif 145 < tourLength <= 300:
			num_islands = 4
			migration = 50 
			nbhoodLevel = 3 
			λopt = 0
			convCounts = 25
		else:
			num_islands = 1
			migration = 100 # not used, therefore set to high vaue
			nbhoodLevel = 5 
			λopt = 1
			convCounts = 15

	# INITIALIZATION

		# SPLIT INTO ISLANDS

		populations = []
		populationMixed = [] # Every island population together
		for isl in range(num_islands):     
			populations.append(initializeKMeans(tsp, λ, λopt))    

		#diversity = [0] * num_islands
		trapped = [0] * num_islands
		means = [0] * num_islands
		bests = [0] * num_islands
		bestSols = [Individual(tsp)] * num_islands
		convergence = False
		conv = 0
		bestObjective = 0

	# EVOLUTION

		iter = 0
		while( not convergence):

		# Empty population mixed 
			populationMixed = []

		# FOR EACH ISLAND

			island = 0
			for population in populations:
				offspring = []

			# RECOMBINATION

				for ind in range(μ):
					# Selection
					parent1 = selection(tsp, population)
					parent2 = selection(tsp, population)
					# Perform different crossovers and mutations for each island to increase diversity
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

					if island % 3 == 0 and np.random.rand() < probLS:
						offspring[ind * 2].order = local_search(tsp, offspring[ind * 2].order)  
						offspring[ind * 2 + 1].order = local_search(tsp, offspring[ind * 2 + 1].order)  
					elif np.random.rand() < probLS:
						offspring[ind].order = local_search(tsp, offspring[ind].order)

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
					# select individuals
					inm1 = selection(tsp, populations[isl])
					inm2 = selection(tsp, populations[isl])

					# add individuals to current island
					offspring.extend([inm1, inm2])

			# ELIMINATION

				population = elimination(population, offspring, tsp)    	    

			# LOCAL SEARCH OVER BEST SOLUTIONS

				fitnesses = [fitness(tsp, ind.order) for ind in population]
				if iter % localSearch == 0 and iter != 0 and trapped[island] < 5:
				# Find best solution from neighborhood at level nbhoodLevel
					bestInd = fitnesses.index(min(fitnesses))
					for ii in range(nbhoodLevel):
						population[bestInd].order = local_search(tsp, population[bestInd].order)    	        	   
				elif trapped[island] >= 5:
				# if population is trapped in local optimum apply randomized improvement in local search
					for ind in population:
						if np.random.rand() < 0.75:
							ind.order = local_search(tsp, ind.order, randomize=True)    	
			
			# CHECK IF TRAPPED IN LOCAL SPACE  

				fitnesses = [fitness(tsp, ind.order) for ind in population]
				# Count number of times that best objective value remains unvaried
				if (bests[island] == min(fitnesses)):
					trapped[island] += 1
				else:        
					trapped[island] = 0       

			# UPDATE NEW POPULATION

				bests[island] = min(fitnesses)
				bestSols[island] = population[fitnesses.index(bests[island])]
				means[island] = statistics.mean(fitnesses)
				populations[island] = population
				populationMixed.extend(population) # Unifiy islands
				#diversity[island] = statistics.mean(fitnesses) - bests[island]
				'''if iter % 5 == 0:
					print(f'---------------------------------  ITERATION {iter}, ISLAND {island} ---------------------------------')  
					print("Mean fitness:  ", f'{statistics.mean(fitnesses):.5f}', '\t', "Best fitness:  ", f'{bests[island]:.5f}',"\t", 'Diversity:', f'{diversity[island]:.5f}')'''
				island += 1				
			
		# REPORT HISTORICAL DATA
			
			meanObjective = statistics.mean(means)
			# Count number of times that total best objective value remains unvaried
			if (int(bestObjective) == int(min(bests))):
				conv += 1
			else:
				conv = 0
			
			bestObjective = min(bests)
			bestSolution = bestSols[bests.index(bestObjective)].order	
			# Check termination conditions
			if num_islands > 1: convergence = (bests.count(bests[0]) == len(bests) and trapped.count(5) == 3) or conv == convCounts				
			else: convergence = conv == convCounts				
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break
			iter += 1

		return 0	

class TSP:
	def __init__(self, matrix):
		self.vertices = [n for n in range(matrix[0].size)]  # vertices from 0 to N
		inf = float('inf')
		if np.max(matrix) == inf:
		  # infinite weights must be changed to finite values
		  matrix = np.where(matrix==inf,1e6, matrix)
		  self.distanceMatrix = matrix
		else:
		  self.distanceMatrix = matrix

class Individual:
	def __init__(self, tsp):
		self.order = np.random.permutation(tsp.vertices)
		# Define Self-adaptative discrete probability range for mutation rates 
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

def initializeKMeans(tsp, λ, λopt): 
	population = []  
	# Initialize enhanced individuals 
	for ii in range(λopt):
		seed = create_seed_ind(tsp)
		# Apply local search over first enhanced individuals
		seed.order = local_search(tsp, seed.order) 
		population.append(seed) 
	# Random initialize the other individuals
	for ii in range(λopt,λ):
		ind = Individual(tsp)
		# Apply local search over first individuals
		ind.order = local_search(tsp, ind.order) 
		population.insert(ii,ind)
	return population

def create_seed_ind(tsp):

  # TRANSFORM MATRIX INTO 2D SPACE

  matrix = tsp.distanceMatrix
  inf = float('inf')
  if np.max(matrix) == inf:
    # infinite weights must be changed to finite values
    matrix = np.where(matrix==inf,1e6, matrix)
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

  	# Compute cluster solution 

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
    # Remove actual cluster from dict   
    centersDict.pop(previous)

    # Append tour solution of the computed cluster

    firstSol.extend(clusterSol)       


  # CREATE FIRST INDIVIDUAL OF POPULATION

  firstInd = Individual(tsp)  
  firstInd.order = np.array(firstSol)

  return firstInd

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
	# Switch mutation every iteration
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
  if i < j:  
    child.order = order_crossover(parent1.order, parent2.order, i, j)
  else:  
    child.order = order_crossover(parent1.order, parent2.order, j, i)
  # Set child parameters  
  if parent1.α == parent2.α:
  	child.α = parent1.α
  elif np.random.rand() < 0.1:
  	# Small probability of setting the mutation rate randomly
  	child.α = random.sample(list(np.linspace(0.5/len(tsp.vertices), 2.5/len(tsp.vertices), 20)), 1)[0]
  else:
  	# Choose mutation rate from parents
  	child.α = random.sample([parent1.α, parent2.α], 1)[0]  
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
  return child

def pmx(parent1,parent2, start, stop):
	child = np.array([None]*len(parent1))
	# Copy parent1 slice from first parent
	child[start:stop] = parent1[start:stop]
	# Map the same slice in parent parent2 to child using indices from parent parent1
	for ind,x in enumerate(parent2[start:stop]):
		ind += start
		if x not in child:
			while child[ind] != None:
				ind = np.where(parent2 == parent1[ind])[0]
			child[ind] = x
	# Copy over the rest from parent parent2
	for ind,x in enumerate(child):
		if x == None:
			child[ind] = parent2[ind]
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

def selection(tsp, population):
  # Tournament selection of 2 individuals selected randomly
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

def local_search(tsp, order, randomize=False):
    min_change = 0
    num_cities = len(order)
    if randomize:
    	min_i = random.sample(list(np.arange(num_cities - 2)), 1)[0] 
    	min_j = random.sample(list(np.arange(min_i + 2, num_cities)), 1)[0]
    	order[min_i+1:min_j+1] = order[min_i+1:min_j+1][::-1]
    	return order
    # Find the best move
    for i in range(num_cities - 2):
        for j in range(i + 2, num_cities - 1):
            change = tsp.distanceMatrix[order[i], order[j]] + tsp.distanceMatrix[order[i+1], order[j+1]] - tsp.distanceMatrix[order[i], order[i+1]] - tsp.distanceMatrix[order[j], order[j+1]]
            if change < min_change:
                min_change = change
                min_i, min_j = i, j                  
    # Update tour with best move
    if min_change < 0:
        order[min_i+1:min_j+1] = order[min_i+1:min_j+1][::-1]
    return order 