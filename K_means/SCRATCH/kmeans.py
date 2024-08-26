import random
import numpy as np
class KMeans:
    def __init__(self,n_clusters=2,max_iterations=100):
        self.n_clusters=n_clusters
        self.max_iterations=max_iterations
        self.centroids=None
    def fit_predict(self,X):
        # print(X.shape[0])
        np.random.seed(0)  # For reproducibility
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        # centroid=random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroids=X[random_indices]
        # print(self.assign_clusters(X))
        # print(self.centroids)

        for i in range(self.max_iterations):
            cluster_group=self.assign_clusters(X)
            #1.assign clusters
            old_centroids=self.centroids 
            #2.move centroids
            self.centroids=self.move_centroids(X,cluster_group)
            #3.Check finish
            if (old_centroids==self.centroids).all():
                break

        return cluster_group
    def assign_clusters(self,X,):
        cluster_group=[]
       
        for row in X:
             distances=[]
             for centroid in self.centroids:
               distances.append( np.sqrt(np.dot(row-centroid,row-centroid)))
             min_distance=min(distances)
             index_position=distances.index(min_distance)
             cluster_group.append(index_position)
        return np.array(cluster_group)
    def move_centroids(self,X,cluster_group):
        new_centroids=[]
        cluster_type=np.unique(cluster_group)
        for type in cluster_type:
           new_centroids.append( X[cluster_group==type].mean(axis=0))
        return np.array(new_centroids)

            

