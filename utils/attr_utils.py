import snap
import numpy as np
import pandas as pd
import random
from w_degree_utils import *

def getEgoAttr(UGraph, node_num, attributes, directed = True):
    egoDeg = np.zeros((node_num,))
    egoOutDeg = np.zeros((node_num,))
    egoInDeg = np.zeros((node_num,))
    egoConn = np.zeros((node_num,))
    avgNeighDeg = np.zeros((node_num,))
    avgNeighInDeg = np.zeros((node_num,))
    avgNeighOutDeg = np.zeros((node_num,))


    for NI in UGraph.Nodes():
        thisNID = NI.GetId()
        NIdegree = attributes['Degree'][thisNID]
        if NIdegree == 0:
            print thisNID, 'degree = 0!'
        InNodes = []
        OutNodes = []

        if directed:
            for Id in NI.GetInEdges():
                InNodes.append(Id)

        for Id in NI.GetOutEdges():
            OutNodes.append(Id)
        EgoNodes = set(InNodes+OutNodes+[NI.GetId()])

        egoID = 0
        egoOD = 0
        neighIDsum = 0
        neighODsum = 0
        egoconn = 0
        for Id in InNodes+OutNodes:
            ego_NI = UGraph.GetNI(Id)

            if directed:
                for IID in ego_NI.GetInEdges():
                    neighIDsum += 1
                    if IID not in EgoNodes:
                        egoID += 1
                    else:
                        egoconn += 1

            for OID in ego_NI.GetOutEdges():
                neighODsum += 1
                if OID not in EgoNodes:
                    egoOD += 1
                else:
                    egoconn += 1


        egoDeg[thisNID] = egoID + egoOD
        egoInDeg[thisNID] = egoID
        egoOutDeg[thisNID] = egoOD
        avgNeighDeg[thisNID] = (neighIDsum+neighODsum)/float(NIdegree)
        avgNeighInDeg[thisNID] = neighIDsum/float(NIdegree)
        avgNeighOutDeg[thisNID] = neighODsum/float(NIdegree)
        egoConn[thisNID] = (egoconn+NIdegree)/float(NIdegree+1)

    attributes['EgonetDegree'] = egoDeg
    attributes['AvgNeighborDeg'] = avgNeighDeg
    attributes['EgonetConnectivity'] = egoConn

    if directed:
        attributes['EgonetInDegree'] = egoInDeg
        attributes['EgonetOutDegree'] = egoOutDeg
        attributes['AvgNeighborInDeg'] = avgNeighInDeg
        attributes['AvgNeighborOutDeg'] = avgNeighOutDeg


def getUndirAttribute(filename, node_num, weighted = None, param = 1.0):
    UGraph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1)
    # UGraph = snap.LoadPajek(snap.PUNGraph, filename + '.paj')
    # or node_num
    attributeNames = ['Graph', 'Id', 'Degree', 'NodeBetweennessCentrality', 
                                       'PageRank', 'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity']
    if weighted:
        attributeNames += ['WeightedDegree', 'EgoWeightedDegree', 'AvgWeightedNeighborDeg', 'EgonetWeightedConnectivity']
    # attributeNames = ['Graph', 'Id', 'Degree', 'NodeBetweennessCentrality', 
                              #          'FarnessCentrality', 'PageRank', 'NodeEccentricity',
                              #          'EgonetDegree', 'AvgNeighborDeg', 'EgonetConnectivity'])
    attributes = pd.DataFrame(np.zeros((node_num, len(attributeNames))), columns =  attributeNames)
                              
    attributes['Graph'] = [filename.split('/')[-1].split('.')[0]] * node_num #node_num
    # Degree
    attributes['Id'] = range(0, node_num) #???????????????? 1, +1?????
    degree = np.zeros((node_num,))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(UGraph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] = degree

    getEgoAttr(UGraph, node_num, attributes, directed=False)

    if weighted:
        df = getWeightedDegree(filename, node_num, attributes, directed=False)
        getWeightedEgoAttr(UGraph, node_num, attributes, df, directed=False)
    # Farness Centrality, Node Eccentricity
    # farCentr = np.zeros((node_num,))
    # nodeEcc = np.zeros((node_num,))
    # for NI in UGraph.Nodes():
    #     farCentr[NI.GetId()] = snap.GetFarnessCentr(UGraph, NI.GetId())
    #     nodeEcc[NI.GetId()] = snap.GetNodeEcc(UGraph, NI.GetId(), False)
    # attributes['FarnessCentrality'] = farCentr
    # attributes['NodeEccentricity'] = nodeEcc

    # Betweenness Centrality
    betCentr = np.zeros((node_num,))
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(UGraph, Nodes, Edges, param)
    for node in Nodes:
        betCentr[node] = Nodes[node]
    attributes['NodeBetweennessCentrality'] = betCentr

    # PageRank
    pgRank = np.zeros((node_num,))
    PRankH = snap.TIntFltH()
    snap.GetPageRank(UGraph, PRankH)
    for item in PRankH:
        pgRank[item] = PRankH[item]
    attributes['PageRank'] = pgRank

    return attributes

def getDirAttribute(filename, node_num, weighted = None, param = 1.0):
    Graph = snap.LoadEdgeList(snap.PNGraph, filename, 0, 1)
    # Graph = snap.LoadPajek(snap.PNGraph, filename + '.paj')
    
    attributeNames = ['Graph', 'Id', 'Degree', 'InDegree', 'OutDegree', 'NodeBetweennessCentrality', 
                      'PageRank', 
                      # 'FarnessCentrality', 'HubsScore', 'AuthoritiesScore', 'NodeEccentricity',
                      'EgonetDegree', 'EgonetInDegree', 'EgonetOutDegree',
                      'AvgNeighborDeg', 'AvgNeighborInDeg', 'AvgNeighborOutDeg','EgonetConnectivity']
    if weighted:
        attributeNames += ['WeightedDegree', 'WeightedInDegree', 'WeightedOutDegree', 'EgoWeightedDegree', 'AvgWeightedNeighborDeg', 'EgonetWeightedConnectivity'\
        , 'EgoWeightedInDegree', 'EgoWeightedOutDegree', 'AvgWeightedNeighborInDeg', 'AvgWeightedNeighborOutDeg']

    attributes = pd.DataFrame(np.zeros((node_num, len(attributeNames))), columns=attributeNames)
    
    attributes['Graph'] = [filename.split('/')[-1].split('.')[0]] * node_num
    attributes['Id'] = range(0, node_num)
    
    # Degree
    degree = np.zeros((node_num,))
    InDegV = snap.TIntPrV()
    snap.GetNodeInDegV(Graph, InDegV)
    for item in InDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] += degree
    attributes['InDegree'] = degree
    
    degree = np.zeros((node_num,))
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(Graph, OutDegV)
    for item in OutDegV:
        degree[item.GetVal1()] = item.GetVal2()
    attributes['Degree'] += degree
    attributes['OutDegree'] = degree
    
    getEgoAttr(Graph, node_num, attributes)

    # attributes['Degree'] /= node_num
    # attributes['InDegree'] /= node_num
    # attributes['OutDegree'] /= node_num

    if weighted:
        df = getWeightedDegree(filename, node_num, attributes, directed=True)
        getWeightedEgoAttr(UGraph, node_num, attributes, df, directed=True)

    # Degree, Closeness, Farness Centrality, Node Eccentricity
    # farCentr = np.zeros((node_num,))
    # nodeEcc = np.zeros((node_num,))
    # for NI in Graph.Nodes():
    #     farCentr[NI.GetId()] = snap.GetFarnessCentr(Graph, NI.GetId(), True, True)
    #     nodeEcc[NI.GetId()] = snap.GetNodeEcc(Graph, NI.GetId(), True)
    # attributes['FarnessCentrality'] = farCentr
    # attributes['NodeEccentricity'] = nodeEcc

    # Betweenness Centrality
    betCentr = np.zeros((node_num,))
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(Graph, Nodes, Edges, param, True)
    for node in Nodes:
        betCentr[node] = Nodes[node]
    attributes['NodeBetweennessCentrality'] = betCentr

    # PageRank
    pgRank = np.zeros((node_num,))
    PRankH = snap.TIntFltH()
    snap.GetPageRank(Graph, PRankH)
    for item in PRankH:
        pgRank[item] = PRankH[item]
    attributes['PageRank'] = pgRank

    # Hubs, Authorities score 
    # hubs = np.zeros((node_num,))
    # auth = np.zeros((node_num,))
    # NIdHubH = snap.TIntFltH()
    # NIdAuthH = snap.TIntFltH()
    # snap.GetHits(Graph, NIdHubH, NIdAuthH)
    # for item in NIdHubH:
    #     hubs[item] = NIdHubH[item]
    # for item in NIdAuthH:
    #     auth[item] = NIdAuthH[item]
    # attributes['HubsScore'] = hubs
    # attributes['AuthoritiesScore'] = auth

    return attributes

def addNodeAttribute(structAttributes, nodeAttributeNames = None, nodeAttributeValues = None, P = None, noise_level = None):
    if len(nodeAttributeNames) > 0 and len(nodeAttributeValues) > 0:
        if noise_level:
            visited = set()
            m, n = np.array(nodeAttributeValues).shape
            for _ in range(int(m * n * noise_level)):
                add1, add2 = np.random.choice(m), np.random.choice(n)
                while ((add1, add2) in visited):
                    add1, add2 = np.random.choice(m), np.random.choice(n)
                nodeAttributeValues[add1][add2] = (nodeAttributeValues[add1][add2] + np.random.choice([1, 2])) % n + 1
        if P is not None:
            nodeAttributeValues = P.dot(np.array(nodeAttributeValues))
            nodeAttributes = pd.DataFrame(nodeAttributeValues.astype(int), columns = nodeAttributeNames)
        else:
            nodeAttributes = pd.DataFrame(nodeAttributeValues, columns = nodeAttributeNames)
        nodeAttributeNames = []
        for col in nodeAttributes.columns:
            onehot = pd.get_dummies(nodeAttributes[col], prefix = col)
            nodeAttributeNames += list(onehot.columns)
            structAttributes = pd.concat([structAttributes, onehot], axis = 1)
    return structAttributes, nodeAttributeNames

def addEdgeAttribute(structAttributes, edgeAttributessName = None, edgeAttributessValue = None, P = None, noise_level = None):
    if len(edgeAttributessName) > 0 and len(edgeAttributessValue) > 0:
        if noise_level:
            visited = set()
            m, n = np.array(edgeAttributessValue).shape
            for _ in range(int(m * n * noise_level)):
                add1, add2 = np.random.choice(m), np.random.choice(n)
                while ((add1, add2) in visited):
                    add1, add2 = np.random.choice(m), np.random.choice(n)
                edgeAttributessValue[add1][add2] = edgeAttributessValue[add1][add2] + np.random.choice([1, 2])
        if P is not None:
            edgeAttributessValue = P.dot(np.array(edgeAttributessValue))
            edgeAttributes = pd.DataFrame(edgeAttributessValue.astype(int), columns = edgeAttributessName)
        else:
            edgeAttributes = pd.DataFrame(edgeAttributessValue, columns = edgeAttributessName)
        
        structAttributes = pd.concat([structAttributes, edgeAttributes], axis = 1)
    return structAttributes



