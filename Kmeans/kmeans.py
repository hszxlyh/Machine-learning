import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    centers=[]
    centers.append(generator.randint(0,n))
    total=[]
    for i in range(1,n_cluster):
        index = centers[i-1]
        distance=np.sum((x-x[index])**2,axis=1)
        if total==[]:
            total=distance.reshape((1,n))
        else:
            distance=distance.reshape((1,n))
            total=np.concatenate((total,distance), axis=0)
        near_cluster=np.min(total,axis=0)

        #new_cluster=near_cluster.argmax()
        new_cluster=np.argmax(near_cluster)
        centers.append(new_cluster)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        iter=0
        J_old=0
        y=np.zeros(N)
        centroids = x[self.centers]
        while(iter<self.max_iter):
            J_new=0
            distance= np.sum((x-np.expand_dims(centroids,axis=1))**2,axis=2)
            y=np.argmin(distance,axis=0)
            value_y=np.min(distance,axis=0)
            J_new=np.sum(value_y)

            # for i in range(0, N):
            #     distance=np.sqrt( np.sum((centroids-x[i])**2,axis=1) )
            #     value_min=np.min(distance)
            #     index_min=np.argmin(distance)
            #     y[i]=index_min
            #     J_new+=value_min
            if np.abs(J_new-J_old)<self.e:
                iter+=1
                break
            J_old=J_new

            uniq_y=np.array(np.unique(y),dtype='int32')
            for j in range(0,len(uniq_y)):
                centroids[uniq_y[j]]= np.mean(x[np.where(y==uniq_y[j])],axis=0)
            iter+=1


        self.max_iter=iter


        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        iter = 0
        J_old = 0
        ycenter = np.zeros(N)
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        centroids = x[self.centers]
        while (iter < self.max_iter):
            J_new = 0
            distance = np.sum((x - np.expand_dims(centroids, axis=1)) ** 2, axis=2)
            y = np.argmin(distance, axis=0)
            value_y = np.min(distance, axis=0)
            J_new = np.sum(value_y)
            if np.abs(J_new - J_old) < self.e:
                iter += 1
                break
            uniq_y = np.array(np.unique(ycenter), dtype='int32')

            for j in range(0, len(uniq_y)):
                centroids[uniq_y[j]] = np.mean(x[np.where(ycenter == uniq_y[j])], axis=0)
            iter += 1

        #----classify
        centroid_labels=np.zeros(self.n_cluster)
        uniq_y = np.array(np.unique(ycenter), dtype='int32')
        for one_center in uniq_y:
            seq,counts=np.unique(y[np.where(ycenter==one_center)],return_counts=True)
            label=np.argmax(counts)
            centroid_labels[one_center]=seq[label]


        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels=np.zeros(N)
        for i in range(0,N):
            distance=np.sum((self.centroids-x[i])**2,axis=1)
            index_min = np.argmin(distance)
            labels[i]=self.centroid_labels[index_min]
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    new_im=np.zeros(image.shape)
    dim_x,dim_y,rgb=image.shape
    for i in range(0,dim_x):
        for j in range(0,dim_y):
            distance=np.sum((image[i,j]-code_vectors)**2,axis=1)
            index=np.argmin(distance)
            new_im[i,j]=code_vectors[index]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

