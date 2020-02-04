# [ Imports ]
# [ -Python ]
from collections import defaultdict
import heapq as hp
import random
# [ -Third Party ]
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def generateCosineBuckets(attributes, cols):
    attr = attributes.drop(['Graph', 'Id'], axis=1).values
    randMatrix = np.random.normal(size=(attr.shape[1], cols))
    #randMatrix = np.multiply(((randMatrix - 0.5) * 2).T, 1 / np.median(attributes, axis=0)).T
    randMatrix = np.multiply(((randMatrix - 0.5) * 2).T, 1 / np.mean(attr, axis=0)).T
    #randMatrix = (randMatrix - 0.5) * 2
    signMatrix = attr.dot(randMatrix)
    signMatrix[signMatrix > 0] = 1
    signMatrix[signMatrix < 0] = 0
    hashMatrix = signMatrix.dot([2**i for i in range(cols)])
    dic = defaultdict(list)
    for i in range(len(hashMatrix)):
        dic[hashMatrix[i]].append((attributes.iloc[i]['Graph'], attributes.iloc[i]['Id']))
    return dic


def generateEuclideanBuckets(attributes, bin_wid):
    attr = attributes.drop(['Graph', 'Id'], axis=1).values
    randVec = np.random.normal(size=(attr.shape[1],))
    randVec = np.multiply((randVec).T, 10 / np.mean(attr, axis=0)).T
    bias = random.uniform(0, bin_wid)
    
    hashVec = (attr.dot(randVec)+bias)/bin_wid
    hashVec = hashVec.astype(int)
    dic = defaultdict(list)
    for i in range((len(hashVec))):
        dic[hashVec[i]].append((attributes.iloc[i]['Graph'], attributes.iloc[i]['Id']))
    return dic


def selectAndCombine(A, B, cols=None):
    if cols is not None:
        return A[cols + ['Graph', 'Id']].append(B[cols + ['Graph', 'Id']], ignore_index=True)
        #return np.vstack((A[cols].values(), B[cols].values()))
    else:
        #return np.vstack((A.values(), B.values()))
        return A.append(B, ignore_index=True)


def selectAndCombineMulti(graph_attrs, cols=None):
    graphs = graph_attrs.keys()
    stacked_attr = pd.DataFrame()\
    # if cols is not None:
    #     stacked_attr = graph_attrs[graphs[0]][cols + ['Graph', 'Id']]
    # else:
    #     stacked_attr = graph_attrs[graphs[0]]
    for i in range(len(graphs)):
        if cols is not None:
            stacked_attr = stacked_attr.append(graph_attrs[graphs[i]][cols + ['Graph', 'Id']], ignore_index=True)
        else:
            stacked_attr = stacked_attr.append(graph_attrs[graphs[i]], ignore_index=True)
    return stacked_attr

def cos_sim(v1, v2, scaling=None):
    # try:
    if scaling is None:
        scaling = np.ones((len(v1),))
    if sum(v1) == 0 or sum(v2) == 0:
        return 0
    v1 = np.multiply(v1, 1/scaling)
    v2 = np.multiply(v2, 1/scaling)

    return v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def KL_sim(distribution1, distribution2):
    v1, bin1 = np.histogram(distribution1, 30)
    v2, bin2 = np.histogram(distribution2, 30)
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    v1 = v1 / sum(v1)
    v2 = v2 / sum(v2) 
    return np.sum(np.where((v1 != 0) & (v2 !=0), v1 * np.log(v1 / v2), 0), axis = 0)

def Euclidean_sim(v1,v2, scaling=None):
    if scaling is None:
        scaling = np.ones((len(v1, )))
    assert (len(v1) == len(v2)), "Dimension is different"
    v1 = np.multiply(v1, 1 / scaling)
    v2 = np.multiply(v2, 1 / scaling)
    eucDis = sum((v1 - v2) ** 2) ** 0.5
    return 1/(1+eucDis)

def computeSparseMatchingMat2(attributesA, attributesB, pair_count_dict, LSHType, threshold=1):
    combineAB = selectAndCombine(attributesA, attributesB)
    combineAB = combineAB.values()
    matching_matrix = lil_matrix((len(attributesA), len(attributesB)))
    scale = np.mean(combineAB[:,2:], axis=0)
    pair_computed = 0
    if LSHType == 'Cosine':               
        for pair, count in pair_count_dict.items():
            if count >= threshold:
                pair_computed += 1
                matching_matrix[pair[0], pair[1]] = cos_sim(combineAB[pair[0]][2:],
                    combineAB[pair[1]+len(attributesA)][2:],scaling=scale)*count
    elif LSHType == 'Euclidean':
        for pair, count in pair_count_dict.items():
            if count >= threshold:
                pair_computed += 1
                matching_matrix[pair[0], pair[1]] = Euclidean_sim(combineAB[pair[0]][2:],
                    combineAB[pair[1]+len(attributesA)][2:],scaling=scale)*count
    matching_matrix = matching_matrix.tocsr()
    matching_matrix = normalize(matching_matrix, norm='l1', axis=1)
    return matching_matrix, pair_computed

def computeSparseMatchingMat(attributesA, attributesB, pair_count_dict, LSHType, threshold):
    combineAB = selectAndCombine(attributesA, attributesB)
    combineAB = combineAB.values
    matching_matrix = lil_matrix((len(attributesA), len(attributesB)))
    scale = np.mean(combineAB[:,2:], axis=0)
    node_pairs = defaultdict(list)
    pair_computed = 0

    for pair, count in pair_count_dict.iteritems():
        hp.heappush(node_pairs[pair[0]], (-1 * count, pair[1]))   # small to large based on negative count

    if LSHType == 'Cosine':               
        for row in node_pairs.keys():
            for _ in range(min(len(node_pairs[row]), int(attributesA.shape[0]*threshold))):
                pair_computed += 1
                neg_count, col = hp.heappop(node_pairs[row])
                matching_matrix[row, col] = cos_sim(
                    combineAB[row][2:],
                    combineAB[col+len(attributesA)][2:], scaling=scale
                ) * (-neg_count)
    elif LSHType == 'Euclidean':
        for row in node_pairs.keys():
            for _ in range(min(len(node_pairs[row]), int(attributesA.shape[0]*threshold))):
                pair_computed += 1
                neg_count, col = hp.heappop(node_pairs[row])
                matching_matrix[row, col] = cos_sim(
                    combineAB[row][2:],
                    combineAB[col+len(attributesA)][2:], scaling=scale
                ) * (-neg_count)
    matching_matrix = matching_matrix.tocsr()
    matching_matrix = normalize(matching_matrix, norm='l1', axis=1)
    return matching_matrix, pair_computed


def computeWholeSimMat(attributesA, attributesB, LSHType):
    combineAB = selectAndCombine(attributesA, attributesB)
    combineAB = combineAB.values
    sim_vec = []
    scale = np.mean(combineAB[:, 2:], axis=0)
    if LSHType == 'Cosine':
        for j in range(len(attributesA)):
            vec = [cos_sim(combineAB[j, 2:], combineAB[len(attributesA)+i, 2:], scale) for i in range(len(attributesB)) ]
            sim_vec.append(vec)
    elif LSHType == 'Euclidean':
        for j in range(len(attributesA)):
            vec = [Euclidean_sim(combineAB[j, 2:], combineAB[len(attributesA)+i, 2:], scale) for i in range(len(attributesB)) ]
            sim_vec.append(vec)

    return csr_matrix(sim_vec)


def combineBucketsBySumMulti(buckets, stacked_attrs, graphs, center_id, reweight=True):
    pair_count_dict = defaultdict(lambda : defaultdict(int))
    for bucket in buckets:
        if len(bucket)>1:
            sorted_bucks = sorted(bucket.items(), key=lambda item: len(item[1]))
            if len(sorted_bucks[-1][1]) > len(stacked_attrs.index) * 3 / 4:
                bucket = dict(sorted_bucks[:-1])
            for buck, collisions in bucket.items(): # collisions = [(Graph, Id)]
                if len(collisions) <= 1:
                    continue
                A_idx = stacked_attrs[(stacked_attrs['Graph'] == center_id)
                    & (stacked_attrs['Id'].isin([c[1] for c in collisions if c[0]==center_id]))]
                if len(collisions) == len(A_idx) or len(A_idx) == 0:    # We don't want all in A
                    continue

                for g in graphs:
                    if g == center_id:
                        continue
                    B_idx = stacked_attrs[(stacked_attrs['Graph'] == g)\
                        & (stacked_attrs['Id'].isin([c[1] for c in collisions if c[0]==g]))]
                    for aid in A_idx.index.values:
                        for bid in B_idx.index.values:
                            if reweight:
                                # experimental
                                pair_count_dict[g][(stacked_attrs['Id'][aid], stacked_attrs['Id'][bid])] += 1.0/len(collisions)*len(stacked_attrs.index)
                            else:
                                pair_count_dict[g][(stacked_attrs['Id'][aid], stacked_attrs['Id'][bid])] += 1.0
    return pair_count_dict


def sparseRank(matching_matrix, P1=None, P2=None, printing=False):
    if P1 is not None and P2 is not None:
        if P1.shape[0] > P2.shape[1]:
            dim = P1.shape[0]
            new_matching = csr_matrix((matching_matrix.data, matching_matrix.nonzero()), shape=(dim, dim))
            new_P2 = P2
        else:
            dim = P2.shape[1]
            new_P2 = csr_matrix((P2.data, P2.nonzero()), shape=(dim, dim))
            new_matching = matching_matrix
    else:
        return None, None
    matching_matrix = P1.T.dot(new_matching).dot(new_P2)

    n, d = matching_matrix.shape
    ranking = np.zeros(min(n, d))
    correct_match = np.zeros(min(n, d))
    sorted_row = defaultdict(list)

    matching_matrix = matching_matrix.tocoo() # For .row and .col
    tuples = zip(matching_matrix.row, matching_matrix.col, matching_matrix.data)
    rank = sorted(tuples, key=lambda x: (x[0], x[2]), reverse=True)
    
    if printing:
        sorted_row = defaultdict(list)
        for r in rank:
            sorted_row[r[0]].append((r[1], r[2]))
        print('sorted_row: {}'.format(sorted_row))

    rank = [(pair[0], pair[1], pair[2]) for pair in rank]
    # Dictionary for each node to other nodes sorted by score
    for r in rank:
        sorted_row[r[0]].append(r[1])

    # Find position of same index
    matching_matrix = matching_matrix.tocsr() # For [i, i]
    for i in range(min(n, d)):
        if i in sorted_row and matching_matrix[i, i] != 0:
            ranking[i] = 1.0 / (sorted_row[i].index(i) + 1)
            correct_match[i] = (i == sorted_row[i][0]) # max matching score at node i
    return ranking, correct_match

def sparseArgmaxMatch(matching_matrix, attributesA, attributesB, P = None):
    if P is not None:
        matching_matrix = matching_matrix.dot(P)
    score =[]
    for i in range(matching_matrix.shape[0]):
        score.append(attributesB['Id'][matching_matrix[i].toarray().argsort()[-1]] == attributesA['Id'][i])
    return score


