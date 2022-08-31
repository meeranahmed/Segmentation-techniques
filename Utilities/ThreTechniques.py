
import numpy as np 
from cv2 import cv2
import matplotlib.pyplot as plt


def new_threshold(gray_image, Threshold):

    # Get Background Array, Consisting of All Pixels With Intensity Lower Than The Given Threshold
    new_background = gray_image[np.where(gray_image < Threshold)]
    # Get Foreground Array, Consisting of All Pixels With Intensity Higher Than The Given Threshold
    new_foreground = gray_image[np.where(gray_image > Threshold)]

    new_background_mean = np.mean(new_background)
    new_foreground_mean = np.mean(new_foreground)
    # Calculate Optimal Threshold
    OptimalThreshold = (new_background_mean + new_foreground_mean) / 2
    return OptimalThreshold

def optimal_thresholding(gray_image):

     # Maximum number of rows and cols for image
    max_x = gray_image.shape[1] - 1
    max_y = gray_image.shape[0] - 1

    first_corner = int(gray_image[0, 0])
    second_corner = int(gray_image[0, max_x])
    third_corner = int(gray_image[max_y, 0])
    forth_corner= int(gray_image[max_x, max_y])

    # Mean Value of Background Intensity, Calculated From The Four Corner Pixels
    background_mean = ( first_corner + second_corner + third_corner + forth_corner ) / 4
    Sum = 0
    Length = 0

    # Loop To Calculate Mean Value of Foreground Intensity
    for i in range(0, gray_image.shape[1]):
        for j in range(0, gray_image.shape[0]):
            # Skip The Four Corner Pixels
            if not ((i == 0 and j == 0) or (i == max_x and j == 0) or (i == 0 and j == max_y) or (i == max_x and j == max_y)):
                Sum += gray_image[j, i]
                Length += 1
    foreground_mean = Sum / Length

    OldThreshold = (background_mean + foreground_mean) / 2
    NewThreshold = new_threshold(gray_image, OldThreshold)

    # Iterate untill the old and new threshold is equal
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = new_threshold(gray_image, OldThreshold)
    
    return NewThreshold


def otsu_thresholding(gray_image):
    HistValues = plt.hist(gray_image.ravel(), 256)[0]
    # print(hist)
    background, foreground = np.split(HistValues,[1])

    within_variance = []
    between_variance = []
    d = 0 
    for i in range(len(HistValues)):
        background, foreground = np.split(HistValues,[i])
        c1 = np.sum(background)/(gray_image.shape[0]* gray_image.shape[1])
        c2 = np.sum(foreground)/(gray_image.shape[0]*gray_image.shape[1])

        background_mean = np.sum([ intensity*frequency for intensity,frequency in enumerate(background)])/np.sum(background)
        background_mean = np.nan_to_num(background_mean)
        foreground_mean = np.sum([ (intensity + d)*(frequency) for intensity,frequency in enumerate(foreground)])/np.sum(foreground)
        
        background_variance = np.sum([(intensity - background_mean)**2*frequency for intensity,frequency in enumerate(background)])/np.sum(background)
        background_variance = np.nan_to_num(background_variance)
        foreground_variance = np.sum([(((intensity + d - foreground_mean)*(intensity + d - foreground_mean))*frequency) for intensity,frequency in enumerate(foreground)])/np.sum(foreground)

        d = d +1
        within_variance.append((c1*background_variance) + (c2*foreground_variance))
        between_variance.append(c1*c2*(background_mean-foreground_mean)*(background_mean-foreground_mean))

    min =np.argmin(within_variance)
    max=np.argmax(background_variance)
    return min
   


def spectral_threshold(image):
    blur = cv2.GaussianBlur(image,(5,5),0)
    hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    print("sum_hist",np.sum(hist))   #65536
    print("len_hist",len(hist))      #256
    hist /= float(np.sum(hist)) 
    BetweenClassVarsList = np.zeros((256, 256))
    for bar1 in range(len(hist)):

        for bar2 in range(bar1, len(hist)):
            ForegroundLevels = []
            BackgroundLevels = []
            MidgroundLevels = []
            ForegroundHist = []
            BackgroundHist = []
            MidgroundHist = []
            for level, value in enumerate(hist):
                if level < bar1:
                    BackgroundLevels.append(level)
                    BackgroundHist.append(value)
                elif level > bar1 and level < bar2:
                    MidgroundLevels.append(level)
                    MidgroundHist.append(value)
                else:
                    ForegroundLevels.append(level)
                    ForegroundHist.append(value)
            
            FWeights = np.sum(ForegroundHist) / float(np.sum(hist))
            BWeights = np.sum(BackgroundHist) / float(np.sum(hist))
            MWeights = np.sum(MidgroundHist) / float(np.sum(hist))
            FMean = np.sum(np.multiply(ForegroundHist, ForegroundLevels)) / float(np.sum(ForegroundHist))
            BMean = np.sum(np.multiply(BackgroundHist, BackgroundLevels)) / float(np.sum(BackgroundHist))
            MMean = np.sum(np.multiply(MidgroundHist, MidgroundLevels)) / float(np.sum(MidgroundHist))
            BetClsVar = FWeights * BWeights * np.square(BMean - FMean) + \
                                                FWeights * MWeights * np.square(FMean - MMean) + \
                                                    BWeights * MWeights * np.square(BMean - MMean)
            BetweenClassVarsList[bar1, bar2] = BetClsVar
        max_value = np.nanmax(BetweenClassVarsList)
    threshold = np.where(BetweenClassVarsList == max_value)
    return threshold






