# [ Imports ]
# [ -Python ]
from collections import defaultdict
from pathlib2 import Path
import sys
# [ -Third Party ]
import numpy as np
import numpy.linalg
import scipy.spatial.distance
from scipy import stats
from scipy.sparse import identity
# [ -Project ]
from utils.lsh_utils import KL_sim, cos_sim
from utils.io_sparse_utils import loadSparseGraph, removeIsolatedSparse, writeSparseToFile


# A should be sparse matrix
# Adding noise based on A, return multiple sparse matrix
def permuteMultiSparse(A, number, graph_type, level, is_perm=True, weighted_noise=None):
	m, n = A.get_shape()
	multi_graph_w_permutation = []
	permutation = []
	B = A.copy()
	noise = [(k, v) for k, v in zip(B.nonzero()[0], B.nonzero()[1]) if k <= v] # No duplicate edges
	visited = set(noise)
	scipy.random.shuffle(noise)
	noise = noise[: int(len(noise) * level // 2) * number] # total number of noise / 2
	# Noise (edges) for each graph
	multi_noise = [noise[len(noise) * i // number: len(noise) * (i+1) // number]for i in range(number)]
	P = identity(m)
	for n in multi_noise:
		P = identity(m)
		# Dealing with existing edges
		B = B.tolil()
		for i, j in n:
			if weighted_noise:
				B[i, j]  = max(min(np.random.normal(1, weighted_noise), 2), 0) # 0 ~ 2
			else:
				B[i, j] = 0
			if graph_type == 'Undirected':
				B[j, i] = B[i, j]
		# Adding edges
		for _ in range(len(n)):  # Same amount as existing edges 
			add1, add2 = np.random.choice(m), np.random.choice(m)
			while ((add1, add2) in visited or (add2, add1) in visited):
				add1, add2 = np.random.choice(m), np.random.choice(m)
			if weighted_noise:
				B[add1, add2] = max(min(np.random.normal(1, weighted_noise), 2), 0) # 0 ~ 2
			else:
				B[add1, add2] = 1
			if graph_type == 'Undirected':
				B[add2, add1] = B[add1, add2]
			visited.add((add1, add2))

		if is_perm:
			perm = scipy.random.permutation(m)
			P = P.tocsr()[perm, :]

		B = B.tocsr()
		B = P.dot(B).dot(P.T)
		multi_graph_w_permutation.append(B)
		permutation.append(P)
		B = A.copy()


	return multi_graph_w_permutation, permutation

def permuteEdgeMultiSparse(A, number, graph_type, level, is_perm = True):
	m, n = A.get_shape()
	multi_graph_w_permutation = []
	permutation = []
	B = A.copy()
	# Noise (edges) for each graph
	P = identity(m)
	for _ in range(number):
		noise = [(k, v) for k, v in zip(B.nonzero()[0], B.nonzero()[1]) if k <= v]	 # No duplicate edges
		P = identity(m)
		# Dealing with existing edges
		B = B.tolil()
		for i, j in noise:
			B[i, j]  = B[i, j] + np.random.uniform(0, level) # 0 ~ 2
			if graph_type == 'Undirected':
				B[j, i] = B[i, j]
		if is_perm:
			perm = scipy.random.permutation(m)
			P = P.tocsr()[perm, :]
		B = P.dot(B).dot(P.T)

		B = B.tocsr()
		multi_graph_w_permutation.append(B)
		permutation.append(P)
		B = A.copy()

	return multi_graph_w_permutation, permutation



# Load original graph from edge file and create multiple synthetic graphs with noise
# Write sparse matrixes to edge file for later use
def generate_multi_graph_synthetic(filename = None, graph_type = 'Undirected', weighted = False, number = 5, edge_noise_only = False, noise_level = 0.02, weighted_noise = None, is_perm = True):
	path = 'metadata/multigraph/'
	graph_info = {} # {graph name: sparse adjacency matrix}
	perm_info = {} # {graph name: permutation} lenth =  number + 1 
	if filename:
		A = loadSparseGraph(filename, graph_type, weighted)
	else:
		raise RuntimeError("Need an input file")
	# Remove Isolated nodes in A
	A, rest_idx = removeIsolatedSparse(A)
	if edge_noise_only:
		multi_graph_w_permutation, permutation = permuteEdgeMultiSparse(A, number, graph_type, level = noise_level, is_perm = is_perm)
	else:
		multi_graph_w_permutation, permutation = permuteMultiSparse(A, number, graph_type, level = noise_level, weighted_noise = weighted_noise, is_perm = is_perm)
	Path(path + graph_type).mkdir(parents=True, exist_ok=True)
	writeSparseToFile(path + graph_type + '/M0.edges', A)
	graph_info['M0'] = A
	perm_info['M0'] = identity(A.get_shape()[0], format='csr')
	for i, gp in enumerate(zip(multi_graph_w_permutation, permutation)):
		g, p = gp
		writeSparseToFile(path + graph_type + '/M' + str(i+1) + '.edges', g)
		graph_info['M'+str(i+1)] = g
		perm_info['M'+str(i+1)] = p

	return graph_info, perm_info, path + graph_type

def get_graph_signature(attributes):
	signature = []
	""" Extract features: Degree, EgonetDegree, Avg Egonet Neighbor, Egonet Connectivity, Clustering Coefficient  """
	for i in range(2, len(attributes.columns)): 
		# if i == 2 or i == 6:
		#  	continue
		feature = attributes.iloc[:, i]  
		# median
		md = np.median(feature)
		# mean
		mn = np.mean(feature)
		# std
		std_dev = np.std(feature)
		# skew
		skew = stats.skew(feature)
		# kurtosis
		krt = stats.kurtosis(feature)
		signature += [md, mn, std_dev, skew, krt]
		#signature += [md, mn, std_dev]
	return signature

def get_multi_graph_signature(graph_attrs):
	multigraph_sig = {}
	for graph, attr in graph_attrs.iteritems():
		multigraph_sig[graph] = get_graph_signature(attr)
	return multigraph_sig

def get_distribution_matrix(aggregations, attributes):
	D = defaultdict(float)
	for a in attributes:
		for g1, attr1 in aggregations.iteritems():
			for g2, attr2 in aggregations.iteritems():
				D[g1] += KL_sim(attr1[a], attr2[a])
	return D

def get_distance(sig1,sig2,type='canberra'):
	if type == 'canberra':
		return scipy.spatial.distance.canberra(sig1, sig2)
	elif type == 'manhattan':
		return numpy.linalg.norm(np.array(sig1) - np.array(sig2), ord=1)
	elif type == 'euclidean':
		return numpy.linalg.norm(np.array(sig1) - np.array(sig2))
	else:
		return cos_sim(sig1,sig2)

def get_distance_matrix_and_order(multigraph, distance = 'canberra'):
	D = defaultdict(float)
	for g1, attr1 in multigraph.iteritems():
		for g2, attr2 in multigraph.iteritems():
			D[g1] += get_distance(attr1, attr2, distance)
	return D

def find_center(multigraph, distance_type='canberra'):
	"""
	rtype: string
	"""
	D = get_distance_matrix_and_order(multigraph, distance_type)
	min_dist = sys.maxint
	center = None
	for g, dist in D.iteritems():
		if min_dist >= dist:
			min_dist = dist
			center = g
	return center
	
if __name__ == '__main__':
	GraphType = 'Undirected'
	path = 'metadata/multigraph/Undirected'
	multi_graphs = generate_multi_graph_synthetic('facebook/0.edges')
	graph_attrs = {}
	attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
	'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']



