from lsh_utils import *
from io_sparse_utils import *
import numpy as np
import numpy.linalg
import pandas as pd
import scipy.spatial.distance
from scipy import stats
from collections import defaultdict
from scipy.sparse import identity
import os
import sys

# A should be sparse matrix
# Adding noise based on A, return multiple sparse matrix
def permuteMultiSparse(A, number, graph_type, level, is_perm = True, weighted_noise = None):
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
				B[i, j]  = max(min(np.random.normal(B[i, j], weighted_noise), B[i, j] + 1), B[i, j] - 1) # 0 ~ 2
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
	writeSparseToFile(path + graph_type + '/M0.edges', A)
	# writeSparseToFile(path + graph_type + '/M0', A)
	graph_info['M0'] = A
	perm_info['M0'] = identity(A.get_shape()[0])
	for i, gp in enumerate(zip(multi_graph_w_permutation, permutation)):
		g, p = gp
		writeSparseToFile(path + graph_type + '/M' + str(i+1) + '.edges', g)
		# writeSparseToFile(path + graph_type + '/M' + str(i+1), g)
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

def get_multi_graph_signature(graph_type = 'Undirected', graph_attrs = None):
	multigraph_sig = {}
	aggregations = {}
	if not graph_attrs:
		path = 'metadata/multigraph/'
		for filename in os.listdir(path + graph_type):
			if not filename.startswith('.'):
				aggregations[filename] = get_graph_feature(path, filename)
	else:
		aggregations = graph_attrs
	for graph, attr in aggregations.iteritems():
		multigraph_sig[graph] = get_graph_signature(attr)
	return multigraph_sig

def get_distribution_matrix(aggregations, attributes):
	m = len(aggregations)
	D = defaultdict(float)
	att = {}
	# attributes = aggregations[0].columns[2:]
	# attributes = ['Degree']
	for a in attributes:
		# for i in range(len(aggregations) - 1):
		# 	for j in range(i + 1, len(aggregations)):
		# 		D[i][j] = KL_sim(aggregations[i][a], aggregations[j][a])
		for g1, attr1 in aggregations.iteritems():
			for g2, attr2 in aggregations.iteritems():
				D[g1] += KL_sim(attr1[a], attr2[a])
		# D = D + D.T
		# att[a] = D
		# D = np.zeros((m, m))
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

def get_distance_matrix_and_order(multigraph, check_center = True, distance = 'canberra'):
	# m = multigraph.keys()
	D = defaultdict(float)
	# if check_center:
	# 	m.remove('center.edges')
	# 	m = ['center.edges'] + m
	# for i, g1 in enumerate(m):
	# 	for j, g2 in enumerate(m):
	# 		if i <= j:
	# 			D[i][j] = get_distance(multigraph[g1], multigraph[g2], distance) 
	for g1, attr1 in multigraph.iteritems():
		for g2, attr2 in multigraph.iteritems():
			D[g1] += get_distance(attr1, attr2, distance)
	# D = D + D.T 
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
			center  = g
	return center
	
if __name__ == '__main__':
	GraphType = 'Undirected'
	path = 'metadata/multigraph/Undirected'
	multi_graphs = generate_multi_graph_synthetic('facebook/0.edges')
	graph_attrs = {}
	attributes = ['Degree', 'NodeBetweennessCentrality', 'PageRank', 
	'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
	# attributes = ['Degree', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
	# node_num = multi_graphs['M0'].get_shape()[0] # m of (m, n)
	# for key in multi_graphs.keys():
	# 	print key
	# 	attributesA = getUndirAttribute(path + '/' + key, node_num)
	# 	# TODO: handle when permutation possible
	# 	with open(path + '/attributes'+key, 'w') as f:
	# 		for index, row in attributesA.iterrows():
	# 			f.write(str(attributesA.ix[index]))
	# 	graph_attrs[key] = attributesA[['Graph', 'Id']+attributes]
	#multigraph_sig = get_multi_graph_signature('Undirected', graph_attrs)
	#D = get_distance_matrix_and_order(multigraph_sig)
	# graph_signatures = get_distribution_matrix(graph_attrs, attributes)
	# print graph_signatures


