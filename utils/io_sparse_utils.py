# [ Imports ]
# [ -Third Party ]
from scipy.sparse import lil_matrix
from scipy.sparse import identity
import numpy as np
import scipy

def loadSparseGraph(fname, graph_type = 'Undirected', weighted = False):
	nodes = []
	A_size = 0
	with open(fname) as f:
		for line in f:
			pair = line.strip().split()
			A_size = max(int(pair[0]), int(pair[1]), A_size)
			if weighted:
				nodes.append((int(pair[0]), int(pair[1]), float(pair[2])))
			else:
				nodes.append((int(pair[0]), int(pair[1]), 1))
	A_size = A_size + 1 # Node Id starts from 0
	A = lil_matrix((A_size, A_size))

	for n in nodes:
		A[n[0], n[1]] = n[2]
		if graph_type == 'Undirected':
			A[n[1], n[0]] = n[2]
	A = A.tocsr()
	return A

def removeIsolatedSparse(A):
	A = A.tocsr()
	rest_bool = ((A.sum(axis=0) != 0).tolist() or (A.sum(axis=1) != 0).tolist())[0] # 2d to 1d
	rest_idx = [i for i in range(len(rest_bool)) if rest_bool[i]]
	A = A[rest_idx, :]
	A = A[:, rest_idx]
	return A, rest_idx

def permuteSparse(A, is_perm = False, has_noise = False, weighted_noise = None, level = 0.05):
	m, n = A.get_shape()
	perm = scipy.random.permutation(m)

	P = identity(m)
	if is_perm:
		P = P.tocsr()[perm, :]
	B = P.dot(A).dot(P.T)
	# Only flip existinf edges
	if has_noise:
		# Flipping existing edges
		noise = [(k, v) for k, v in zip(B.nonzero()[0], B.nonzero()[1]) if k <= v]
		visited = set(noise)  # Remove duplicate edges in undirected
		scipy.random.shuffle(noise)
		noise = noise[: int(len(noise[0]) * level)]
		B = B.tolil()
		for pair in noise:
			if weighted_noise:
				B[pair[1], pair[0]] = B[pair[0], pair[1]] = max(min(np.random.normal(1, weighted_noise), 2), 0) # 0 ~ 2
			else:
				B[pair[0], pair[1]] = 0
				B[pair[1], pair[0]] = 0
		# Adding edges, should not be visited before
		for _ in range(int(m*m*level)):
			add1, add2 = np.random.choice(m), np.random.choice(m)
			while ((add1, add2) in visited or (add2, add1) in visited):
				add1, add2 = np.random.choice(m), np.random.choice(m)
			if weighted_noise:
				B[add1, add2] = B[add2, add1] = max(min(np.random.normal(1, weighted_noise), 2), 0)
			else:
				B[add1, add2] = 1
				B[add2, add1] = 1
			visited.add((add1, add2))
		B = B.tocsr()

	return B, P

# B is a sparse matrix
def writeSparseToFile(fname, B):
	weighted_edges = [zip(B.nonzero()[0], B.nonzero()[1], B.data)][0]  # list of node1 -> node2 with weight
	with open(fname, 'w') as f:
		for e1, e2, w in weighted_edges:
			f.write(str(e1)+" "+str(e2)+" "+str(w)+"\n")
	f.close()

def loadNodeFeature(fname):
	nodeFeaturesValue = []
	nodeFeaturesName =[]
	with open(fname) as f:
		nodeFeaturesName = f.readline().strip().split()
		for line in f:
			v = line.strip().split()
			nodeFeaturesValue.append([int(i) for i in v])
	return nodeFeaturesValue, nodeFeaturesName

def loadEdgeFeature(fname):
	edgeFeaturesValue = []
	edgeFeaturesName =[]	
	with open(fname) as f:
		edgeFeaturesName = f.readline().strip().split()
		for line in f:
			line = line.strip().split()
			edgeFeaturesValue.append([int(i) for i in line])
	return edgeFeaturesValue, edgeFeaturesName












