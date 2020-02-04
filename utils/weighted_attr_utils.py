# [ Imports ]
# [ -Third Party ]
import numpy as np
import pandas as pd


def getWeightedEgoAttr(UGraph, node_num, attributes, df, directed=True):
    egoDeg = np.zeros((node_num,))
    egoOutDeg = np.zeros((node_num,))
    egoInDeg = np.zeros((node_num,))
    egoConn = np.zeros((node_num,))
    avgNeighDeg = np.zeros((node_num,))
    avgNeighInDeg = np.zeros((node_num,))
    avgNeighOutDeg = np.zeros((node_num,))

    wnodes = {}
    wnodes = dict(zip(zip(df[0], df[1]), df[2]))

    for NI in UGraph.Nodes():
        thisNID = NI.GetId()
        NIdegree = attributes['WeightedDegree'][thisNID]
        if NIdegree == 0:
            print(thisNID, 'degree = 0!')
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
                    neighIDsum += wnodes[(IID, Id)]
                    if IID not in EgoNodes:
                        egoID += wnodes[(IID, Id)]
                    else:
                        egoconn += wnodes[(IID, Id)]

            for OID in ego_NI.GetOutEdges():
                neighODsum += wnodes[(Id, OID)]
                if OID not in EgoNodes:
                    egoOD += wnodes[(Id, OID)]
                else:
                    egoconn += wnodes[(Id, OID)]


        egoDeg[thisNID] = egoID + egoOD
        egoInDeg[thisNID] = egoID
        egoOutDeg[thisNID] = egoOD
        avgNeighDeg[thisNID] = (neighIDsum+neighODsum)/float(NIdegree)
        avgNeighInDeg[thisNID] = neighIDsum/float(NIdegree)
        avgNeighOutDeg[thisNID] = neighODsum/float(NIdegree)
        egoConn[thisNID] = (egoconn+NIdegree)/float(NIdegree+1)

    attributes['EgoWeightedDegree'] = egoDeg
    attributes['AvgWeightedNeighborDeg'] = avgNeighDeg
    attributes['EgonetWeightedConnectivity'] = egoConn

    if directed:
        attributes['EgoWeightedInDegree'] = egoInDeg
        attributes['EgoWeightedOutDegree'] = egoOutDeg
        attributes['AvgWeightedNeighborInDeg'] = avgNeighInDeg
        attributes['AvgWeightedNeighborOutDeg'] = avgNeighOutDeg


# Should be two-way
def getWeightedDegree(filename, node_num, attributes, directed = True):
    df = pd.read_csv(filename, header = None, sep = ' ')
    wdegree = np.zeros((node_num, ))
    df_d = df.groupby([0])[2].sum()
    # Out degree
    for i in df_d.index:
        wdegree[i] = df_d[i]
    attributes['WeightedDegree'] = wdegree
    if directed:       
        attributes['WeightedOutDegree'] = wdegree
        # In degree
        df_d = df.groupby([1])[2].sum()
        wdegree = np.zeros((node_num, ))
        for i in df_d.index:
            wdegree[i] = df_d[i]
        attributes['WeightedInDegree'] = wdegree
        attributes['WeightedDegree'] += wdegree

    return df