# [ Imports ]
# [ -Python ]
from pathlib2 import Path
import os.path
import pickle
import time
import sys
import warnings
# [ -Third Party ]
import pandas as pd
import numpy as np
# [ -Project ]
from utils.lsh_utils import \
    computeWholeSimMat, generateCosineBuckets, generateEuclideanBuckets,\
    selectAndCombineMulti, combineBucketsBySumMulti, sparseRank, computeSparseMatchingMat


class HashAlign:
    def __init__(self, fname):
        self.sim_matrix = {}
        self.best_ranking = {}
        self.best_correct_match = {}
        self.fname = 'exp_result/' + fname

    def experiment(self, df, filename, bandNumber=4, LSHType='Euclidean',
                   loop_num=1, cos_num_plane=50, euc_width=3, compute_sim=False,
                   threshold=0.2, reweight=True):
        ## debug
        np.seterr(all='raise')
        warnings.filterwarnings('error')

        # Load all necessary data
        metadata = {}
        centers = []

        # Load synthetic graph information
        with open('./private_data/' + filename + '/metadata') as f:
            for line in f:
                line = line.strip().split()
                metadata[line[0]] = line[1]

        # Check multiple graphs
        if metadata['number'] >= 1:
            with open('./private_data/' + filename + '/centers') as f:
                for line in f:
                    centers.append(line.strip().split()[0])
                f.close()
        else:
            raise RuntimeError("Need two graphs to align")

        # Load all graph attributes
        graph_attrs = pickle.load(open('./private_data/' + filename + '/attributes.pkl', 'rb'))
        graph_perm = pickle.load(open('./private_data/' + filename + '/permutations.pkl', 'rb'))
        multi_graphs = pickle.load(open('./private_data/' + filename + '/multi_graphs.pkl', 'rb'))
        if Path('./private_data/' + filename + '/node_label.pkl').exists():
            node_label = pickle.load(open('./private_data/' + filename + '/node_label.pkl', 'rb'))
        # Load attributes name
        attributes = []
        with open('./private_data/' + filename + '/attributes') as f:
            for line in f:
                attributes.append(line.strip().split()[0])

        for center_id in centers:
            rank_score = 0
            rank_score_upper = 0
            correct_score = 0
            correct_score_upper = 0
            pairs_computed = 0
            avg_derived_rank = 0

            start_sim = time.time()
            if compute_sim:
                for g in graph_attrs.keys():
                    if (center_id, g) not in self.sim_matrix and g != center_id:
                        print('!!! computed self.sim_matrix !!!')
                        self.sim_matrix[(center_id, g)] = computeWholeSimMat(graph_attrs[center_id], graph_attrs[g],
                                                                             LSHType)
                    if (center_id, g) not in self.best_ranking and g != center_id:
                        self.best_ranking[(center_id, g)], self.best_correct_match[(center_id, g)] = sparseRank(
                            self.sim_matrix[(center_id, g)], graph_perm[center_id], graph_perm[g])
            end_sim = time.time()
            print('sim_time: ' + str(end_sim - start_sim))

            start_matching = time.time()

            for i in range(loop_num):
                ## ------------ generate buckets ------------ ##
                band_all = list(attributes)
                np.random.shuffle(band_all)
                randomBand = [band_all[i * len(band_all) / bandNumber: (i + 1) * len(band_all) / bandNumber] for i in
                              range(bandNumber)]

                buckets = []

                if LSHType == 'Cosine':
                    for band in randomBand:
                        buckets.append(generateCosineBuckets(selectAndCombineMulti(graph_attrs, band), cos_num_plane))


                elif LSHType == 'Euclidean':
                    for band in randomBand:
                        buckets.append(generateEuclideanBuckets(selectAndCombineMulti(graph_attrs, band), euc_width))

                ## ---------- end generate buckets ----------- ##

                stacked_attrs = selectAndCombineMulti(graph_attrs)
                pair_count_dict = combineBucketsBySumMulti(buckets, stacked_attrs[['Graph', 'Id']], graph_attrs.keys(),
                                                           center_id, reweight)

                matching_matrix = {}
                this_pair_computed = {}
                Ranking = {}
                correctMatch = {}

                for g in pair_count_dict.keys():
                    if g == center_id:
                        continue
                    matching_matrix[g], this_pair_computed[g] \
                        = computeSparseMatchingMat(graph_attrs[center_id], graph_attrs[g], pair_count_dict[g], LSHType,
                                                   threshold)
                    Ranking[g], correctMatch[g] = sparseRank(matching_matrix[g], graph_perm[center_id], graph_perm[g])
                    rank_score += sum(Ranking[g]) / len(Ranking[g])
                    correct_score += sum(correctMatch[g]) / float(len(correctMatch[g]))
                    if not compute_sim:
                        self.best_ranking[(center_id, g)] = Ranking[g]
                        self.best_correct_match[(center_id, g)] = correctMatch[g]
                        rank_score_upper += 0
                        correct_score_upper += 0
                    else:
                        rank_score_upper += sum(self.best_ranking[(center_id, g)]) / len(
                            self.best_ranking[(center_id, g)])
                        correct_score_upper += sum(self.best_correct_match[(center_id, g)]) / float(
                            len(self.best_correct_match[(center_id, g)]))

                    pairs_computed += this_pair_computed[g] / float(
                        matching_matrix[g].shape[0] * matching_matrix[g].shape[1])

                    print("==========================================================")
                    print(filename + ' ' + g + ', center:' + center_id + ', center_dist: ' + metadata[
                        'center_distance'])
                    print("GraphType = " + metadata['graph_type'])
                    print("bandNumber = " + str(bandNumber) + ", LSHType = " + LSHType)
                    print("noise_level = " + metadata['noise_level'] + ", nodeAttributeFile = " + metadata[
                        'node_dir'] + ", threshold = " + str(threshold))
                    print("matching score by ranking: %f" % (sum(Ranking[g]) / len(Ranking[g])))
                    if compute_sim:
                        print("matching score by ranking upper bound: %f" % (
                                sum(self.best_ranking[(center_id, g)]) / len(self.best_ranking[(center_id, g)])))
                    print("matching score by correct match: %f" % (sum(correctMatch[g]) / float(len(correctMatch[g]))))
                    if compute_sim:
                        print(
                            "matching score by correct match upper bound %f" % (
                                sum(self.best_correct_match[(center_id, g)])
                                / float(len(self.best_correct_match[(center_id, g)]))
                            )
                        )
                    print("percentage of pairs computed: %f" % (this_pair_computed[g] / float(
                        matching_matrix[g].shape[0] * matching_matrix[g].shape[1])))

                if int(metadata['number']) > 1:
                    derived_matching_matrix = {}
                    derived_rank = {}
                    non_center = matching_matrix.keys()
                    for i in range(len(non_center)):
                        for j in range(i + 1, len(non_center)):
                            derived_matching_matrix[(non_center[i], non_center[j])] = matching_matrix[
                                non_center[i]].T.dot(matching_matrix[non_center[j]])
                            Ranking, correct_match = sparseRank(derived_matching_matrix[(non_center[i], non_center[j])],
                                                                graph_perm[non_center[i]], graph_perm[non_center[j]])
                            derived_rank[(non_center[i], non_center[j])] = sum(Ranking) / len(Ranking)

                    print('derived rank score: ')
                    print(derived_rank)
                    tmp_avg_derived_rank = sum([v for k, v in derived_rank.iteritems()]) / len(derived_rank)
                    avg_derived_rank += tmp_avg_derived_rank
                    print('avg derived rank score: ' + str(tmp_avg_derived_rank))

            rank_score /= loop_num * len(pair_count_dict.keys())
            rank_score_upper /= loop_num * len(pair_count_dict.keys())
            correct_score /= loop_num * len(pair_count_dict.keys())
            correct_score_upper /= loop_num * len(pair_count_dict.keys())
            pairs_computed /= loop_num * len(pair_count_dict.keys())
            avg_derived_rank /= loop_num
            end_matching = time.time()
            matching_time = end_matching - start_matching
            print("matching_time: {}".format(matching_time))

            df = df.append({'filename': filename
                           , 'nodeAttributeFile': metadata['node_dir']
                           , 'edge_label_dir': metadata['edge_label_dir']
                           , 'noise_level': metadata['noise_level']
                           , 'GraphType': metadata['graph_type']
                           , 'bandNumber': bandNumber
                           , 'LSHType': LSHType
                           , 'cos_num_plane': cos_num_plane
                           , 'euc_width': euc_width
                           , 'threshold': threshold
                           , 'rank_score': rank_score
                           , 'rank_score_upper': rank_score_upper
                           , 'correct_score': correct_score
                           , 'correct_score_upper': correct_score_upper
                           , 'center_id': center_id
                           , 'found_center': metadata['found_center']
                           , 'avg_derived_rank': avg_derived_rank
                           , 'center_dist': metadata['center_distance']
                           , 'pairs_computed': pairs_computed
                           , 'preprocess_time': metadata['preprocess_time']
                           , 'matching_time': matching_time
                            },
                           ignore_index=True)
        return df

    def run(self, band_numbers=[4], cos_num_plane=[25], euc_width=[4], LSHs=['Cosine'],
            folders=['facebook'], thresholds=[0.2]):

        # center_distance_types = ['canberra', 'manhattan', 'euclidean']

        if os.path.isfile(self.fname + '.pkl'):
            with open(self.fname + '.pkl', 'rb') as f:
                df = pickle.load(f)
        else:
            df = pd.DataFrame()

        for fold in folders:
            for band in band_numbers:
                for thres in thresholds:
                    for lsh in LSHs:
                        if lsh == 'Cosine':
                            for c in cos_num_plane:
                                df = self.experiment(df, filename=fold,
                                                     bandNumber=band, LSHType=lsh, cos_num_plane=c, threshold=thres,
                                                     compute_sim=False)
                        else:
                            for e in euc_width:
                                df = self.experiment(df, filename=fold,
                                                     bandNumber=band, LSHType=lsh, euc_width=e, threshold=thres,
                                                     compute_sim=False)
                        Path('exp_result').mkdir(exist_ok=True)
                        pickle.dump(df, open(self.fname + '.pkl', 'wb'))
                        df.to_csv(self.fname + '.csv')
            self.sim_matrix = {}
            self.best_ranking = {}
            self.best_correct_match = {}

if __name__ == '__main__':
    ha_runner = HashAlign(fname=sys.argv[1])
    ha_runner.run()
