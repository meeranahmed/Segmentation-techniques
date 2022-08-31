import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from Utilities.RGB_to_LUV import *


def getFeatureVector(Image, Feature_Dimension=3):
    '''
    Extract feature space according to type of feature 
    inputs:
        Image : the Image itself
        feature : intensity(1D), color(HS) (2D) or color(RGB)(3D)
    return:
        feature vector.
    '''
    m, n = Image.shape[0:2]
    luv_Image= RGB_To_LUV(Image)
    # luv_Image = cv2.cvtColor(Image, cv2.COLOR_RGB2Luv)
    num_points = m*n
    if Feature_Dimension == 1:
        im_space = luv_Image[..., 2]
    elif Feature_Dimension == 2:
        im_space = luv_Image[..., 0:2]
    elif Feature_Dimension == 3:
        im_space = Image
    else:
        exit('Not supported feature')
    feature_vector = np.reshape(im_space, (num_points, Feature_Dimension)).T
    return feature_vector

def getInitialMean(Feature_Vector, UnVisited_Points):
    '''
    Get a random point from feature space as a starting mean
    Return: random mean
    '''
    # Get a random Point
    idx = int(np.round(len(UnVisited_Points) * np.random.rand()))
    # Check boundary condition
    if idx >= len(UnVisited_Points):
        idx -= 1
    return Feature_Vector[:, int(UnVisited_Points[idx])]

def clusterImage(Original_Image, Pixel_Cluster, clusters):
    '''
    Extract results of clustering by assigning the cluster center to all its points and returning back to Image space
    inputs:
        Pixel_Cluster: pixel cluster pair (1xnum_points)
        clusters: cluster feature pair ( Clusters_Num x Feature_Dimension ) 
    Return: 
        Image After Segmentation 
    '''
    m, n = Original_Image.shape[0:2]
    clusters = np.asarray(clusters).T
    Feature_Dimension, Clusters_Num = clusters.shape[0:2]
    clusterd_feature_space = np.zeros(
        (len(Pixel_Cluster), clusters.shape[0])).T
    # Map values to pixels according to its cluster
    for c in range(Clusters_Num):
        idxs = np.where(Pixel_Cluster == c)
        for j in idxs[0]:
            clusterd_feature_space[:, j] = clusters[:, c]
    # Return to Image space
    im_space = np.reshape(clusterd_feature_space.T, (m, n, Feature_Dimension))
    if Feature_Dimension == 1:
        im_space = im_space[..., 0]
        segmented_Image = im_space
    elif Feature_Dimension == 2:
        luv_Image= RGB_To_LUV(Original_Image)
        # luv_Image = cv2.cvtColor(Original_Image, cv2.COLOR_RGB2Luv)
        luv_Image[..., 0:2] = im_space
        luv_Image[..., 2] /= np.max(luv_Image[..., 2])
        segmented_Image = cv2.cvtColor(luv_Image, cv2.COLOR_Luv2RGB)
    else:
        segmented_Image = im_space
    return segmented_Image

def meanShift(Image, Bandwidth, Feature):
    Start_Time = time.time()
    '''
    inputs : 
        Image : to be segmented
        Bandwidth : window radius of in range points
    output : segmented Image & number of clusters
    '''
    feature_vector = getFeatureVector(Image, Feature)
    num_points = feature_vector.shape[1]
    visited_points = np.zeros(num_points)
    threshold = 0.05*Bandwidth
    
    # Initialize an empty list of clusters
    clusters = []
    num_clusters = -1
    not_visited = num_points
    not_visited_Idxs = np.arange(num_points)
    out_vector = -1*np.ones(num_points)

    while not_visited:
        new_mean = getInitialMean(feature_vector, not_visited_Idxs)
        this_cluster_points = np.zeros(num_points)
        while True:
            dist_to_all = np.sqrt(np.sum((feature_vector.T-new_mean)**2, 1)).T
            in_range_points_idxs = np.where(dist_to_all < Bandwidth)
            visited_points[in_range_points_idxs[0]] = 1
            this_cluster_points[in_range_points_idxs[0]] = 1
            old_mean = new_mean
            new_mean = np.sum(feature_vector[:, in_range_points_idxs[0]],1)/in_range_points_idxs[0].shape[0]
            if np.isnan(new_mean[0]):
                break
            if np.sqrt(np.sum((new_mean - old_mean)**2)) < threshold:
                
                # Merge checking with other clusters
                merge_with = -1
                for i in range(num_clusters+1):
                    # Get distance between clusters
                    dist = np.sqrt(np.sum((new_mean - clusters[i])**2))
                    # Merge condition
                    if dist < 0.5 * Bandwidth:
                        merge_with = i
                        break
                if merge_with >= 0:
                    # In case of merge ... Get in between mean and update it to old cluster
                    clusters[merge_with] = 0.5 * \
                        (new_mean + clusters[merge_with])
                    # Mark this cluster point as belongs to cluster we merge with
                    out_vector[np.where(this_cluster_points == 1)] = merge_with
                else:
                    # No merging ... Make a new cluster
                    num_clusters += 1
                    # Add it to our list
                    clusters.append(new_mean)
                    out_vector[np.where(
                        this_cluster_points == 1)] = num_clusters
                break
        not_visited_Idxs = np.array(np.where(visited_points == 0)).T
        not_visited = not_visited_Idxs.shape[0]
        
    # Image Segmentation
    segmented_Image = clusterImage(Image, out_vector, clusters)
    End_Time = time.time()
    # exec_time = abs(End_Time - Start_Time)
    print(f"Execution time is{End_Time - Start_Time} sec")
    return segmented_Image, num_clusters+1

# image = cv2.imread('images/kmean+meanshift.png')
# image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Bandwidth = 0.1*np.max(image)
# segmented_image, num_clusters = meanShift(image, Bandwidth=0.1*np.max(image),Feature=3)
# # plt.figure("Segmented Image")
# # plt.imshow(segmented_image)
# # plt.waitforbuttonpress()
# cv2.imshow('output Image', segmented_image)
# cv2.waitKey(0)

  
