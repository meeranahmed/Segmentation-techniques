import numpy as np
import cv2 as cv
import random
import time
def compute_init_matrix(points) :
    Dis_Mat = [[-1]*len(points) for i in range(0,len(points))]
    for i in range(len(points)) :
        for j in range(i,len(points)) :
            if(j == i) : 
                Dis_Mat[j][i] = -1
                continue
            Dis_Mat[j][i] = compute_Dis(points[i],points[j],i,j)
    return Dis_Mat

def compute_Dis(p1,p2,i,j) :
    p1 = np.append(p1,i)
    p2 = np.append(p2,j)
    return np.sqrt(np.sum(np.square(np.subtract(p1,p2))))

def Min_Dis(matrix) :
    minimum = [1,0]
    for i in range(len(matrix)) :
        for j in range(len(matrix[0])) :
            if((matrix[i][j] == -1)) : continue
            if(matrix[i][j] < matrix[minimum[0]][minimum[1]]) :
                minimum = [i,j]
    return minimum

def segment(image,number_of_clusters = 2) :
    Start_Time = time.time()
    points = image.reshape(image.shape[0] * image.shape[1] , 3)
    dindogram = [[i] for i in range(len(points))]
    if(number_of_clusters > len(points)) : raise Exception("Clusters exceeded points!!")
    Dis_Mat = compute_init_matrix(points)
    while len(dindogram) != number_of_clusters :
        minimum = Min_Dis(Dis_Mat)
        new_cluster = [dindogram[minimum[0]],dindogram[minimum[1]]]
        flat_new_cluster = [item for sublist in new_cluster for item in sublist]
        dindogram.pop(np.max(minimum))
        dindogram[np.min(minimum)] = flat_new_cluster
        Update_Mat(Dis_Mat,minimum[0],minimum[1])
    End_Time = time.time()
    print(f"Execution time of Agglomerative method{End_Time - Start_Time} sec")
    return points,dindogram

def Update_Mat(Dis_Mat,indx1,indx2) :
    maximum_indx = max([indx1,indx2])
    single_link(Dis_Mat,indx1,indx2)
    Dis_Mat.pop(maximum_indx)
    for i in range(len(Dis_Mat)) :
        Dis_Mat[i].pop(maximum_indx)
    

def single_link(Dis_Mat,indx1,indx2) :
    minimum_indx = min([indx1,indx2])
    for i in range(len(Dis_Mat)) :
        if(i == indx1) : continue
        if(i == indx2) : continue
        if(i < indx1) :
            if(i < indx2) :
                distanc_1 = Dis_Mat[indx1][i]
                distanc_2 = Dis_Mat[indx2][i]
                m = min([distanc_1,distanc_2])
                Dis_Mat[minimum_indx][i] = m
            else :
                distanc_1 = Dis_Mat[indx1][i]
                distanc_2 = Dis_Mat[i][indx2]
                m = min([distanc_1,distanc_2])
                if(minimum_indx == indx2) : Dis_Mat[i][minimum_indx] = m
                else : Dis_Mat[i][minimum_indx] = m
                
        else :
            if(i < indx2) :
                distanc_1 = Dis_Mat[i][indx1]
                distanc_2 = Dis_Mat[indx2][i]
                if(minimum_indx == indx1) : Dis_Mat[i][minimum_indx] = m
                else : Dis_Mat[i][minimum_indx] = m
            else :
                distanc_1 = Dis_Mat[i][indx1]
                distanc_2 = Dis_Mat[i][indx2]
                m = min([distanc_1,distanc_2])
                Dis_Mat[i][minimum_indx] = m

def draw(points,dindogram,Image_Before) :
    colors = []
    while len(colors) != len(dindogram):
        color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        if(color not in colors) : colors.append(color)
    for i in range(len(dindogram)) :
        for j in range(len(dindogram[i])) :
            indx = dindogram[i][j]
            points[indx] = colors[i]
    image = points.reshape(Image_Before.shape)
    # filename = "image.png"
    # cv.imwrite(filename,image)
    return image

# Image_Before = cv.imread("./images/AGG_img.png")
# image = np.array(Image_Before)
# points,dindogram = segment(image,15)
# output=draw(points,dindogram,Image_Before)
# cv.imshow('output_image', output)
# cv.waitKey(0)
