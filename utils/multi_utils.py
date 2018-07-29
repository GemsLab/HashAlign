from io_utils import *
from lsh_utils import *
from attr_utils import getEgoAttr
import numpy.linalg
import snap
import pandas as pd
import numpy as np
import scipy.spatial.distance
from scipy import stats
import os

# generate <number> synthetic graphs using <A> with noise=<level>
# noise of different graphs are disjoint
def permuteMultiNoise(A, number, level):
	noise = np.zeros((len(A), len(A)))
	multi_graph_w_permutation = []
	noise_nodes = np.where(np.triu(np.random.choice([0, 1], size=(len(A), len(A)), p=[(100-level * number)/100, level * number /100])))
	noise_nodes = zip(noise_nodes[0], noise_nodes[1])
	np.random.shuffle(noise_nodes)
	multi_noise = [noise_nodes[len(noise_nodes) * i // number: len(noise_nodes) * (i+1) // number]for i in range(number)]
	for n in multi_noise:
		for i, j in n:
			noise[i][j] = 1
		B = (A + noise + noise.T) % 2
		multi_graph_w_permutation.append(B)
		noise = np.zeros((len(A), len(A)))
	return multi_graph_w_permutation

# generate a dict of graph_name:graph from a edge file <filename>
def generate_multi_graph_synthetic(filename = None, graph_type = 'Undirected', number = 5, noise_level = 0.02):
	path = 'metadata/multigraph/'
	# multi_graph_w_permutation = []
	graph_info = {} # {graph name: adjacency matrix}
	if filename:
		A = loadGraph(filename, graph_type)
	elif graph_type == 'Undirected':
		A = np.where(np.triu(np.random.rand(5,5), 1) >= 0.5, 1, 0)
		A += A.T
	else:
		A = np.where(np.triu(np.random.rand(5,5), 1) + np.tril(np.random.rand(5,5), -1) >= 0.5, 1, 0)
		#A = np.where(A - np.triu(A.T) >= 1, 1, 0) 
	A, rest_idx = removeIsolatedNodes(A)
	# for i in range(number):
	# 	multi_graph_w_permutation.append(permuteNoiseMat(A, is_perm = False, has_noise = True, level = 0.05))
	multi_graph_w_permutation = permuteMultiNoise(A, number, level = noise_level)
	writeEdgesToFile(path + graph_type + '/M0.edges', A)
	graph_info['M0.edges'] = A
	for i, g in enumerate(multi_graph_w_permutation):
		writeEdgesToFile(path + graph_type + '/M' + str(i+1) + '.edges', g)
		graph_info['M'+str(i+1)+'.edges'] = g

	return graph_info


def get_node_degree(UGraph, graph_type, attributes):
    degree = np.zeros((UGraph.GetNodes(),))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(UGraph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] = degree

def get_clustering_coeff(UGraph, attributes):
	coeff = np.zeros((UGraph.GetNodes(), ))
	for NI in UGraph.Nodes():
		i = NI.GetId()
		coeff[i] = snap.GetNodeClustCf(UGraph, i)
	attributes['ClusteringCoeff'] = coeff


def get_graph_feature(path, filename, graph_type = 'Undirected'):

	UGraph = snap.LoadEdgeList(snap.PUNGraph, path + graph_type + '/' + filename, 0, 1)
	attributes = pd.DataFrame(np.zeros(shape=(UGraph.GetNodes(), 7)),
		columns=['Graph', 'Id', 'Degree', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity', 'ClusteringCoeff'])
	attributes['Graph'] = [filename] * (UGraph.GetNodes())
	attributes['Id'] = range(1, UGraph.GetNodes()+1)
	# Get node degree
	get_node_degree(UGraph, graph_type, attributes)
	# Get 3 egonet features
	getEgoAttr(UGraph, attributes, directed = False)
	# Get clustering_coeff
	get_clustering_coeff(UGraph, attributes)

	return attributes	

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
	m = multigraph.keys()
	D = np.zeros((len(m), len(m)))
	# if check_center:
	# 	m.remove('center.edges')
	# 	m = ['center.edges'] + m
	for i, g1 in enumerate(m):
		for j, g2 in enumerate(m):
			if i <= j:
				D[i][j] = get_distance(multigraph[g1], multigraph[g2], distance) 
	D = D + D.T 
	return D, m


def find_center(multigraph, distance_type='canberra'):
	"""
	rtype: string
	"""
	D, m = get_distance_matrix_and_order(multigraph, distance_type)
	min_index = np.argmin(sum(D))
	return m[min_index]

if __name__ == '__main__':
	generate_multi_graph_synthetic(filename = 'facebook/0.edges', graph_type = 'Undirected')
	graph_signatures = get_multi_graph_signature()
	print(sum(get_distance_matrix_and_order(graph_signatures)[0]))
	print(find_center(graph_signatures))
