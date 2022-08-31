import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def gray_diff(img, current_point, temp_point):
    return abs(int(img[current_point[0], current_point[1]]) - int(img[temp_point[0], temp_point[1]]))

def connects_selection(p):
    if p != 0:
        connects = [[-1, -1], [0, -1], [1, -1],[1, 0],[1, 1],[0, 1],[-1, 1],[-1, 0]]
    else:
        connects = [[0, -1], [1, 0],[0, 1], [-1, 0]]
    return connects

def fit(img,seeds, thresh, p=1):
    height, width = img.shape
    seed_mark = np.zeros(img.shape)
    seed_list = []
    for seed in seeds:
        seed_list.append(seed)
    label = 1
    connects = connects_selection(p)
    while(len(seed_list) > 0):
        current_pixel = seed_list.pop(0)
        seed_mark[current_pixel[0], current_pixel[1]] = label
        for i in range(8):
            tmpX = current_pixel[0] + connects[i][0]
            tmpY = current_pixel[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue
            grayDiff = gray_diff(img, current_pixel, [tmpX, tmpY])
            if grayDiff < thresh and seed_mark[tmpX, tmpY] == 0:
                seed_mark[tmpX, tmpY] = label
                seed_list.append([tmpX, tmpY])
    return seed_mark
    
# seeds = [[25, 35],[88, 200],[30, 250]]
# # image = cv2.imread("./images/REG_img.png",0)
# image = cv2.imread("./images/01.png",0)
# Start_Time = time.time()
# output_image = fit(image,seeds,6)
# End_Time = time.time()
# print(f"Execution time of region growing method{End_Time - Start_Time} sec")
# cv2.imshow('output_image', output_image)
# cv2.waitKey(0)

# image=plt.imread("./images/REG_img.png",0)
# output_image = fit(image,seeds,6)
# plt.imshow(output_image)



