import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred
    def pruning_compute(self, features,y_labels):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split
        
        ##add
        self.error=0
        self.case=[]
        #self.newdata=0

    #TODO: try to split current node
    def split(self):
        
        if(len(self.features[0])==0):
            self.splittable=False
            return 
        
        temp,freq=np.unique(self.labels,return_counts=True)
        totallen=len(self.labels)
        index_label=0
        dict_label={}
        
        ##....handle with labels without increasing order
        for one_label in temp:
            dict_label[one_label]=index_label
            index_label+=1
        
        
        #print(self.features)
        S=np.dot(-freq/totallen,np.log(freq/totallen))
        
        best_attri_no=0
        best_ig=float('-inf')
        
        
        feat_dim=len(self.features[0])
        for idx_feat in range(0,feat_dim):
            single_feat=np.copy(self.features)[:,idx_feat]
            count_feat=np.unique(single_feat)
            count=len(count_feat)#Number of attribute values
            branches=np.zeros((len(count_feat),self.num_cls))
            for i, val in enumerate(count_feat):
                pick_y=np.copy(self.labels)[np.where(single_feat==val)]
                
                indexj=0
                for j in pick_y:
                    branches[i,dict_label[j]]+=1
                    
            info_gain=Util.Information_Gain(S,branches)
            if(info_gain>best_ig):
                self.dim_split=idx_feat
                best_ig=info_gain
                best_attri_no=count
            elif(info_gain==best_ig and count>best_attri_no ):
                self.dim_split=idx_feat
                best_attri_no=count
        
        select_feat=np.array(self.features)[:,self.dim_split]
        self.feature_uniq_split=np.unique(select_feat).tolist() 
        
        #print(self.feature_uniq_split)
        
        for v in self.feature_uniq_split:
            search_idx=np.where(select_feat==v)
            child_feat=np.delete(np.array(self.features)[search_idx],obj=self.dim_split,axis=1).tolist()
            child_label=np.array(self.labels)[search_idx].tolist()
            childnode=TreeNode(child_feat,child_label,self.num_cls)
            
            
            self.children.append(childnode)
        for each_child in self.children:
            if(each_child.splittable==True):
                each_child.split()
                
        return 
    
        raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
       
        if self.splittable==True:
            if feature[self.dim_split] not in self.feature_uniq_split:
                return self.cls_max
            else:
                child_idx=self.feature_uniq_split.index(feature[self.dim_split])
                new_feat=np.delete(feature,self.dim_split)
                return self.children[child_idx].predict(new_feat)     
        else:    
            return self.cls_max     
        
        raise NotImplementedError
        
    def pruning_traverse(self, feature,y_true):
        # feature: List[any]
        # return: int
       
        if self.splittable==True:
            if feature[self.dim_split] not in self.feature_uniq_split:
                print("oooooooooooooo")
                self.case.append(y_true)
                if(self.cls_max!=y_true):
                    self.error+=1
                    
                return 
                #return self.cls_max
            else:
                child_idx=self.feature_uniq_split.index(feature[self.dim_split])
                new_feat=np.delete(feature,self.dim_split)
                
                self.case.append(y_true)
                self.children[child_idx].pruning_traverse(new_feat,y_true)
                return 
                
                #return self.children[child_idx].pruning_traverse(new_feat,y_true)
            
        else:
            self.case.append(y_true)
            if(self.cls_max!=y_true):
                self.error+=1
            return 
        raise NotImplementedError
        
#     def pruning_clsmax(self):
#         if(len(self.case)==0):
#             return 
#         count_max=0
#         for one_label in np.unique(self.case):
#             if self.case.count(one_label) > count_max:
#                 count_max = self.case.count(one_label)
#                 self.cls_max = one_label
#         for child in self.children:
#             child.pruning_clsmax()
        
    
    def  pruning_geterror(self):
        # get parent's error and case....
        
        
             
        if(len(self.children)==0):
            return self.error
        else:
            
            sum=0
            for child in self.children:
                sum+=child.pruning_geterror()
            
            #his own original error???
            self.error=sum+self.error
            return self.error
        
    def cut(self):
        if(len(self.children)==0):
            return 
        olderror=self.error
        newerror=0
        
        lencase=len(self.case)
        for i in range(0,lencase):
            if(self.case[i]!=self.cls_max):
                newerror+=1
        
              
        
        #get max class
#         localmax=0
#         setcls_max=0
#         for label in np.unique(self.case):
#             if self.case.count(label) > localmax:
#                 localmax = self.case.count(label)
#                 setcls_max=label
        
#         for i in range(len(self.case)):
#             if(self.case[i]!=setcls_max):
#                 newerror+=1
#         #self.newdata=newerror
        
        #newerror<=olderror
        if(newerror<=olderror):
            self.children=[]
            self.splittable=False
            
            #self.cls_max=setcls_max-----CAN NOT  be changed
            
            
            return 
        else:
            for node in self.children:
                node.cut()
            

        
    
    
    
    
    
    
    
    
    
    
