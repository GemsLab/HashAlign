# [ Imports ]
# [ -Project ]
from preprocessing_syn import preprocessing
from HashAlign import HashAlign

if __name__ == '__main__':
	folders = ['facebook']
	euc_widths = {'facebook': [4]}
	cos_num_planes = {'facebook': [25]}
	thresholds = [0.10]
	LSHs = ['Cosine']
	band_numbers = {'facebook': [4]}
	noise_levels = [0.02]
	num_graph = 5
	fname = 'exp_facebook'
	for f in folders:
		for noise_level in noise_levels:
			preprocessing(node_dir='data/'+f+'.nodes', edge_dir='data/'+f+'.edges', save_dir=f, number=num_graph-1, noise_level=noise_level, weighted=False)

			hashalign_runner = HashAlign(fname)
			hashalign_runner.run(
				folders=[f], band_numbers=band_numbers[f], cos_num_plane=cos_num_planes[f],
				LSHs=LSHs, thresholds=thresholds,
			)
