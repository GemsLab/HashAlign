import numpy as np 
from utils.attr_utils import *
from utils.lsh_utils import *
from utils.io_sparse_utils import *
from utils.multi_sparse_utils import *
from utils.baseline_utils import *
import pandas as pd
import os.path
import pickle
import time
from scipy.sparse import csr_matrix

class PureBaseline:
	def __init__(self, fname, baseline_type):
		self.fname = 'exp_result/' + fname
		self.metadata = {}
		self.centers = []
		self.graph_attrs = {}
		self.graph_perm = {}
		self.multi_graphs = {}
		self.sim_matrix = {}
		self.matching_matrix = {}

		self.avg_baseline_score = 0
		self.matching_time = 0
		self.baseline_scores = {}

		self.baseline_type = baseline_type

	def load_data(self, filename):
		# Load synthetic graph information
		with open('./private_data/' + filename + '/metadata') as f:
			for line in f:
				line = line.strip().split()
				self.metadata[line[0]] = line[1]

		# Check multiple graphs
		if self.metadata['number'] >= 1:	
			with open('./private_data/' + filename + '/centers') as f:
				for line in f:
					self.centers.append(line.strip().split()[0])
				f.close()
		else:
			raise RuntimeError("Need two graphs to align")

		node_att_num = int(self.metadata['node_attribute_number'])

		# Load all graph attributes
		self.graph_attrs = pickle.load(open('./private_data/' + filename + '/attributes.pkl', 'rb'))
		if node_att_num > 0:
			for g, att in self.graph_attrs.iteritems():
				cols = list(att.iloc[:, :2])+list(att.iloc[:, -node_att_num:])
				self.graph_attrs[g] = att[cols]
		self.graph_perm = pickle.load(open('./private_data/' + filename + '/permutations.pkl', 'rb'))	
		self.multi_graphs = pickle.load(open('./private_data/' + filename + '/multi_graphs.pkl', 'rb'))
		if int(self.metadata['node_label']) == 1 and os.path.exists('./private_data/' + filename + '/node_label.pkl'):
			self.node_label = pickle.load(open('./private_data/' + filename + '/node_label.pkl', 'rb'))
		else:
			self.node_label = None
			


	def sim_baseline(self, df, filename, LSHType, threshold = 0.2, all_1 = False):
		
		self.load_data(filename)
		node_att_num = int(self.metadata['node_attribute_number'])
		if node_att_num == 0:
			all_1 = True
		start_match = time.time()
		for center_id in self.centers:
			for g in self.graph_attrs.keys():
				if (center_id, g) not in self.sim_matrix and (g, center_id) not in self.sim_matrix and g != center_id:
					if (g, center_id) in self.sim_matrix:
						self.sim_matrix[(center_id, g)] = self.sim_matrix[(g, center_id)]
					else:
						if not all_1:
							print '!!! computed sim_matrix !!!'
							self.sim_matrix[(center_id, g)] = computeWholeSimMat(self.graph_attrs[center_id], self.graph_attrs[g], LSHType)
					if all_1:
						print '!!! use all 1 matrix !!!'
						self.matching_matrix[(center_id, g)] = csr_matrix(np.ones((self.multi_graphs[center_id].shape[0], self.multi_graphs[g].shape[0])))
					else:
						self.matching_matrix[(center_id, g)] = self.filter_sim_to_match(self.sim_matrix[(center_id, g)], threshold)	

			for g in self.multi_graphs.keys():
				if g == center_id:
					continue
				if (g, center_id) not in self.baseline_scores:
					self.baseline_scores[(center_id, g)] = self.get_baseline_score(self.multi_graphs[center_id], self.multi_graphs[g], self.matching_matrix[(center_id, g)]
														, self.graph_perm[center_id], self.graph_perm[g])
				else:
					self.baseline_scores[(center_id, g)] = self.baseline_scores[(g, center_id)]
				self.avg_baseline_score += self.baseline_scores[(center_id, g)]

				print "=========================================================="
				print filename + ' ' + g + 'center: '+center_id
				print "GraphType = " + self.metadata['graph_type'] 
				print "noise_level = " + self.metadata['noise_level'] + ", nodeAttributeFile = " + self.metadata['node_dir']
				self.print_baseline_score(self.baseline_scores[(center_id, g)])
				temp = self.baseline_scores[(center_id, g)]
				# print "netalign score: %f" %(self.baseline_scores[(center_id, g)])
		self.avg_baseline_score /=  (len(self.multi_graphs.keys())**2 - len(self.multi_graphs.keys()) )
		self.matching_time = time.time() - start_match
		print "matching_time: "+str(self.matching_time)

		df = self.append_df(df, LSHType, filename, temp)

		# df = df.append({'filename':filename, 'nodeAttributeFile': self.metadata['node_dir']\
		# 	, 'noise_level':self.metadata['noise_level']\
		# 	, 'avg_netalign_score': self.avg_baseline_score\
		# 	, 'matching_time': self.matching_time\
		# 	}, ignore_index=True)

		return df

	def get_baseline_score(self, A, B, M, Pa, Pb):
		pass
	
	def print_baseline_score(self, baseline_score):
		pass
	def append_df(self, df, filename, baseline_score):
		return

	def filter_sim_to_match(self, sim_matrix, percentage):
		print "[before] sim_matrix non zero %: {}".format(len(sim_matrix.nonzero()[0])/float(sim_matrix.shape[0]**2))
		sim_lil = sim_matrix.tolil()
		def max_n_percent(row_data, row_id, n):
			if not n:
				n = 1
			idx = row_data.argsort()[-n:]
			top_vals = row_data[idx]
			top_ids = row_id[idx]
			return top_vals, top_ids, idx
		for i in xrange(sim_lil.shape[0]):
			d, r = max_n_percent(np.array(sim_lil.data[i])
					, np.array(sim_lil.rows[i]), int(percentage*sim_matrix.shape[0]))[:2]
			sim_lil.data[i]=d.tolist()
			sim_lil.rows[i]=r.tolist()
		sim_matrix = sim_lil.tocsr()
		print "[after] sim_matrix non zero %: {}".format(len(sim_matrix.nonzero()[0])/float(sim_matrix.shape[0]**2))
		return sim_matrix

	def run(self, filename = 'facebook', LSHType = 'Cosine', threshold = 0.2, all_1 = False):
		if os.path.isfile(self.fname+'.pkl'):
			with open(self.fname+'.pkl', 'rb') as f:
				df = pickle.load(f)
		else:
			df = pd.DataFrame()
		df = self.sim_baseline(df, filename = filename, LSHType=LSHType, threshold = threshold, all_1 = all_1)
		pickle.dump(df, open(self.fname+'.pkl','wb'))
		df.to_csv(self.fname+'.csv')

class PureNetAlign(PureBaseline):

	def __init__(self, fname):
		PureBaseline.__init__(self, fname, 'netalign')

	def get_baseline_score(self, A, B, M, Pa, Pb):
		return getNetalignScore(A, B, M, Pa, Pb)[0]

	def print_baseline_score(self, baseline_score):
		print "netalign score: %f" %(baseline_score)

	def append_df(self, df, LSHType, filename, baseline_score):
		df = df.append({'filename':filename
			, 'nodeAttributeFile': self.metadata['node_dir']\
			, 'noise_level':self.metadata['noise_level']\
			, 'LSHType':LSHType\
			, 'netalign_score': baseline_score\
			, 'avg_netalign_score': self.avg_baseline_score\
			, 'matching_time': self.matching_time\
			}, ignore_index=True)
		return df

class PureIsoRank(PureBaseline):

	def __init__(self, fname):
		PureBaseline.__init__(self, fname, 'isorank')

	def get_baseline_score(self, A, B, M, Pa, Pb):
		return getIsoRankScore(A, B, M, Pa, Pb)

	def print_baseline_score(self, baseline_score):
		print "IsoRank score: %f" %(baseline_score)

	def append_df(self, df, LSHType, filename, baseline_score):
		df = df.append({'filename':filename\
			, 'nodeAttributeFile': self.metadata['node_dir']\
			, 'noise_level':self.metadata['noise_level']\
			, 'LSHType':LSHType\
			, 'isorank_score': baseline_score\
			, 'avg_isorank_score': self.avg_baseline_score\
			, 'matching_time': self.matching_time\
			}, ignore_index=True)
		return df

class PureFinal(PureBaseline):

	def __init__(self, fname, edge_label = False):
		PureBaseline.__init__(self, fname, 'final')
		self.edge_label = edge_label

	def get_baseline_score(self, A, B, M, Pa, Pb):
		return getFinalScore(A, B, M, Pa, Pb, node_A = self.node_label, node_B = self.node_label, edge_label = self.edge_label)[0]

	def print_baseline_score(self, baseline_score):
		print "FINAL score: %f" %(baseline_score)

	def append_df(self, df, LSHType, filename, baseline_score):
		df = df.append({'filename':filename
			, 'nodeAttributeFile': self.metadata['node_dir']\
			, 'noise_level':self.metadata['noise_level']\
			, 'LSHType':LSHType\
			, 'final_score': baseline_score\
			, 'avg_final_score': self.avg_baseline_score\
			, 'matching_time': self.matching_time\
			}, ignore_index=True)
		return df


if __name__ == '__main__':
	netalign_runner = PureFinal(sys.argv[1])
	netalign_runner.run(filename = 'DBLP-A')
