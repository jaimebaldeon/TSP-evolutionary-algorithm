import numpy as np 
import sys
import math
import random
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

def two_opt(tsp, order):
    min_change = 0
    num_cities = len(order)
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

def dist(a, b):
    """Return the euclidean distance between cities tour[a] and tour[b]."""
    return np.hypot(coords[tour[a], 0] - coords[tour[b], 0],
                    coords[tour[a], 1] - coords[tour[b], 1])

def local_search(tsp, order):
  selected = order
  bestFit = fitness(tsp, order)
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
      # take neighbour with lowest fitness 
      if fitness(tsp, newOrder) < bestFit:
        bestFit = fitness(tsp, newOrder)
        selected = newOrder
    n+=1  
  return  selected

def local_search_fast(tsp, order):
  selected = order
  bestFit = fitness(tsp, order)
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
      # take neighbour with lowest fitness 
      if fitness(tsp, newOrder) < bestFit:
        bestFit = fitness(tsp, newOrder)
        selected = newOrder
    n+=1
  return  selected

filename = sys.argv[1]
print(f'Solving {filename}... ')
matrix = np.genfromtxt(filename,delimiter=',')
# Create TSP problem class
tsp = TSP(matrix) 

num = 300
p1 = np.arange(int(num))
#p1 = np.random.permutation(order)
print("Parent:")
print(fitness(tsp, p1))
init_t = time.perf_counter()

c1=two_opt(tsp, p1)
#c2=local_search(tsp, p1)

end_t = time.perf_counter()
print('Total time: ',end_t - init_t)
print("Children:")
print(fitness(tsp, c1))
#print(fitness(tsp, c2))
