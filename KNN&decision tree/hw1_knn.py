from __future__ import division, print_function

from typing import List

import numpy as np
import scipy
from collections import Counter
############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features=features
        self.labels=labels
        

        #raise NotImplementedError

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        lentest=len(features)
        labellist=[]
        for i in range(0,lentest):
            k_nodes=self.get_k_neighbors(features[i])
            for ck in range(0,self.k):
                k_nodes[ck]=self.labels[int(k_nodes[ck])]
                
                
            labellist.append(Counter(k_nodes).most_common(1)[0][0])
            
        return labellist     
        #raise NotImplementedError

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        lenfeat=len(self.features)
        dis=[]
        res=[]
        for i in range(lenfeat):
            distance=self.distance_function(point,self.features[i])
            dis.append([distance,i])
        
        dis=np.array(dis)
        
        dis0=dis[:,0]
        dis1=dis[:,1]
        dd=np.lexsort((dis1,dis0))
        dis=dis[dd]
        
#         dis=dis[dis[:,0].argsort()]


        #print(dis)
        for j in range(0,self.k):
            index=dis[j,1]
            res.append(index)
        print(res)
        
        return res
        #raise NotImplementedError


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
