import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sume=0
    totalcase=np.sum(branches)
    row=len(branches)
    col=len(branches[0])
    for i in range(row):
        temptotal=np.sum(branches[i])
        #temp_prop=np.sum(branches[i])/totalcase
        tempsum=0
        for j in range(col):
            if(branches[i][j]==0):
                continue
            else:
                tempsum+=-(branches[i][j]/temptotal)*np.log2(branches[i][j]/temptotal)
       
        sume+=(temptotal/totalcase)*tempsum
    
    return S-sume

    
    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    for i in range(len(X_test)):
        decisionTree.root_node.pruning_traverse(X_test[i],y_test[i])
    
    #decisionTree.root_node.pruning_clsmax()
    temp_error=decisionTree.root_node.pruning_geterror()
    decisionTree.root_node.cut()
    
    return 
    raise NotImplementedError


    
    
# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])
    
    #print("case: ",node.case,"")
    
    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(len(real_labels)):
        if(real_labels[i]==1):
            if(predicted_labels[i]==1):
                tp+=1
            else:
                fn+=1
        else:
            if(predicted_labels[i]==1):
                fp+=1
            else:
                tn+=1
                
    #print(tp,fp,fn,tn)
    return (2*np.dot(real_labels,predicted_labels))/(np.sum(real_labels)+np.sum(predicted_labels))
#     precision=tp/(tp+fp)
#     recall= tp/(tp+fn)
    #raise NotImplementedError

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    
#     point3=[]
#     for i in range(len(point1)):
#         point3.append(point1[i]-point2[i])
    point1=np.array(point1)
    point2=np.array(point2)
    return    np.linalg.norm(point1-point2,2)    
    #raise NotImplementedError


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    point1=np.array(point1)
    point2=np.array(point2)
    return np.dot(point1,point2)
    #raise NotImplementedError


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    
#     point3=[]
#     for i in range(len(point1)):
#         point3.append(point1[i]-point2[i])
    point1=np.array(point1)
    point2=np.array(point2)
    return  -np.exp(-0.5*np.dot(point1-point2,point1-point2) )
    
    #raise NotImplementedError


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    point1=np.array(point1)
    point2=np.array(point2)
    x1=np.dot(point1,point2)
    p1norm=np.linalg.norm(point1,2)
    p2norm=np.linalg.norm(point2,2)
    
    return 1-x1/(p1norm*p2norm)
    #raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    
    print(Xtrain,ytrain,Xval,yval)
    
    model=KNN(1,distance_funcs['euclidean'])
    
    optf1=0
    bestk=-1
    bestfunc=''
    maxk=29
    if(len(Xtrain)<maxk):
        maxk=len(Xtrain)-1
        
    
    for key_func in distance_funcs:
        k=1
        while(k<=maxk):
            model.train(Xtrain,ytrain)
            model.k=k
            model.distance_function=distance_funcs[key_func]
            ypre=model.predict(Xval)
            get_f1=f1_score(yval,ypre)
            
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name=key_func, k=k) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=get_f1))
            print()
            
            if(get_f1>optf1):
                bestk=k
                bestfunc=key_func
                optf1=get_f1
                
            
            k+=2
    print("bestk:  ",bestk,"bestfunc:  ",key_func)
    model.k=bestk
    model.distance_function=distance_funcs[bestfunc]
    return model,bestk,bestfunc
    #raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    model=KNN(1,distance_funcs['euclidean'])
    # initilize
    bestk=1
    bestfunc='euclidean'
    bestscaler='min_max_scale'
    optf1=0
    kmax=29
    if(len(Xtrain)<kmax):
        kmax=len(Xtrain)-1
    
    for scaling_name in scaling_classes:
            
            scaling=scaling_classes[scaling_name]()
            New_Xtrain=scaling.__call__(Xtrain)
            
            New_Xval=scaling.__call__(Xval)
            
            print(scaling_name,New_Xval)
            
            for key_func in distance_funcs:
                k=1
         
                while(k<kmax):
                
                    model.k=k
                    model.distance_function=distance_funcs[key_func]
                    model.train(New_Xtrain,ytrain)
                    
                    
                    ypreval=model.predict(New_Xval)
                
                    get_f1=f1_score(yval,ypreval)
                    if(get_f1>optf1):
                        bestk=k
                        bestfunc=key_func
                        bestscaler=scaling_name
                        optf1=get_f1
                    
                    print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=key_func, scaling_name=scaling_name, k=k) +
                            'valid: {valid_f1_score:.5f}'.format(valid_f1_score=get_f1))
                    
                    print()
                    
                    k+=2
                    
    model.k=bestk
    model.distance_function=distance_funcs[key_func]
    model.scale=scaling_classes[bestscaler]
    print("bestk:  ",bestk,"bestfunc:  ",bestfunc,"bestscale:  ",bestscaler)
   
    
    return model,bestk,bestfunc,bestscaler
    raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        newfeat=np.array(features,dtype='float64')
        for i in range(len(features)):
            norm=np.linalg.norm(features[i],2)
            if(norm==0):
                for j in range(len(newfeat[0])):
                    newfeat[i][j]=0
                continue
            newfeat[i]=np.divide(newfeat[i],norm)
        return newfeat
        
        
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.hascall=False
        self.minarr=np.array([])
        self.maxarr=np.array([])

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        toarr_feature=np.array(features)
        if(self.hascall==False):
            self.minarr=toarr_feature.min(axis=0)
            self.maxarr=toarr_feature.max(axis=0)
            newfeat=np.array(features,dtype='float64')
            for i in range(0,len(features)):
                for j in range(0,len(features[0])):
                    if(self.minarr[j]==self.maxarr[j]):
                        newfeat[i,j]=0
                    else:
                        newfeat[i,j]=(newfeat[i,j]-self.minarr[j])/(self.maxarr[j]-self.minarr[j])
            self.hascall=True
        else:
            newfeat=np.copy(features)
            for i in range(0,len(features)):
                for j in range(0,len(features[0])):
                    if(self.minarr[j]==self.maxarr[j]):
                        newfeat[i,j]=0
                    else:
                        newfeat[i,j]=(newfeat[i,j]-self.minarr[j])/(self.maxarr[j]-self.minarr[j])
        
        return newfeat
        
        
        
        raise NotImplementedError





