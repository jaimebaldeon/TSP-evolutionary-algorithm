import numpy as np 
import sys
import math
import random


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

def pmx_recombination(a,b):
	half = len(a) // 2
	start = random.randint(0, len(a)-half)
	stop = start + half
	return pmx(a,b,start,stop) , pmx(b,a,start,stop)    


filename = sys.argv[1]
print(f'Solving {filename}... ')
matrix = np.genfromtxt(filename,delimiter=',')
# Create TSP problem class
tsp = TSP(matrix) 

num = 100
order = np.arange(int(num))
p1 = np.random.permutation(order)
p2 = np.random.permutation(order)

c,d = pmx_recombination(p1,p2)
print("Parents:")
print(fitness(tsp, p1), p1)
print(fitness(tsp, p2), p2)
print("Children:")
print(fitness(tsp, c),c)
print(fitness(tsp, d),d)
