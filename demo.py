from preprocessing_syn import *
from HashAlign import *
# from pure_baseline import *

if __name__ == '__main__':
	# email, dblp
	folders = ['email','dblp-A']
	LSHs = ['Cosine']
	band_numbers = {'email':[4], 'dblp-A':[2]}
	cos_num_planes = {'email':[50], 'dblp-A':[100]}
	thresholds = [0.20]
	noise_levels = [0.02]
	num_graph = 2
	fname = 'exp_email_dblp-A'
	for f in folders:
		for noise_level in noise_levels:
			preprocessing(edge_dir = 'Data/'+f+'.edges', save_dir = f, number = num_graph-1, noise_level = noise_level
				, node_dir = 'Data/'+f+'.nodes', node_label = True, weighted_noise = 1.0, findcenter = -1)
				
			hashalign_runner = HashAlign(fname)
			hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
					, LSHs = LSHs, thresholds = thresholds)
	# scalability for brain graphs
	folders = ['brain']
	LSHs = ['Cosine']
	band_numbers = {'brain':[2]}
	cos_num_planes = {'brain':[40]}
	thresholds = [0.10]
	nums = [2,4,8,16,32,64]
	fname = 'exp_brain_scalability'
	for f in folders:
		for num in nums:
			preprocessing(edge_dir = 'Data/'+f+'.edges', save_dir = f, number = num-1, noise_level = 0.02
				, weighted_noise = 1.0, weighted = True, findcenter=-1)
			hashalign_runner = HashAlign(fname)
			hashalign_runner.run(folders = [f], band_numbers = band_numbers[f], cos_num_plane = cos_num_planes[f]
                                        , LSHs = LSHs, thresholds = thresholds)
