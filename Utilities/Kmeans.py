import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt

from Utilities.RGB_to_LUV import *


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans():

    def __init__(self, K=5, max_iterations=100):
       
        self.K = K
        self.max_iterations = max_iterations
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):

        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # Optimize clusters
        for _ in range(self.max_iterations):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self.create_clusters(self.centroids)
  
            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)
    
            # check if clusters have changed
            if self.is_converged(centroids_old, self.centroids):
                break

        # Classify samples as the index of their clusters
        return self.get_cluster_labels(self.clusters)
    
    def get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def modelCentroids(self):
        return self.centroids


def main():
    
    img = plt.imread('./images/kmean+meanShift.png')
    # convert to LUV
    # img= RGB_To_LUV(img)
    img = cv.cvtColor(img, cv2.COLOR_RGB2Luv)
    # img = cv.cvtColor(img, cv.COLOR_BGR2LUV)
    # reshape image to points
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    np.seterr(invalid='ignore')
    k = 4
    max_iter = 100
    # run clusters_num-means algorithm
    myModel = KMeans(K=k, max_iterations=max_iter)
    t1 = time.time()
    predictions = myModel.predict(pixel_values)
    t2 = time.time()
    print("\nKMeans (k = {0}) Computation time = {1} sec\n".format(k, (t2 - t1)))
    centers = np.uint8(myModel.modelCentroids())
    predictions = predictions.astype(int)
    # flatten labels and get segmented image
    labels = predictions.flatten()
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(img.shape)
    plt.figure("Segmented Image")
    plt.imshow(segmented_image)
    plt.waitforbuttonpress()
    # cv.imshow('Segmented Image using Kmeans',segmented_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 


    ######### opencv outcome ##########
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # ret,label,center = cv.kmeans(pixel_values,k,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # # Now convert back into uint8, and make original image
    # center = np.uint8(center)
    # im = center[label.flatten()]
    # im = im.reshape((img.shape))
    # cv.imshow('CV.KMEANS',im)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 


if __name__ == '__main__':
    main()