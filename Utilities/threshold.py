import numpy as np
import cv2 
import time
import matplotlib.pyplot as plt
from Utilities.ThreTechniques import spectral_threshold,optimal_thresholding,otsu_thresholding

def Global_threshold(image , thresh_typ = "Optimal"):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_img = np.zeros(image.shape)
    if thresh_typ == "Otsu":
        threshold = otsu_thresholding(image)
        thresh_img = np.uint8(np.where(image > threshold, 255, 0))
    elif thresh_typ == "Optimal":
        threshold = optimal_thresholding(image)
        thresh_img = np.uint8(np.where(image > threshold, 255, 0))
   
    elif thresh_typ=='spect':
        threshold1, threshold2 = spectral_threshold(image)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row, col] > threshold2[0]:
                    thresh_img[row, col] = 255
                elif image[row, col] < threshold1[0]:
                    thresh_img[row, col] = 0
                else:
                    thresh_img[row, col] = 128   
    return thresh_img

def Local_threshold(image, block_size , thresh_typ = 'spect'):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_img = np.copy(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            mask = image[row:min(row+block_size,image.shape[0]),col:min(col+block_size,image.shape[1])]
            thresh_img[row:min(row+block_size,image.shape[0]),col:min(col+block_size,image.shape[1])] = Global_threshold(mask, thresh_typ)
    return thresh_img



# ## optimal image ###
# optimal_img = cv2.imread("./images/Threshold.png")
# optimal_img = cv2.resize(optimal_img,(300,300))
# optimal_img = cv2.cvtColor(optimal_img, cv2.COLOR_BGR2GRAY)
# Start_Time = time.time()
# result1 = Global_threshold(optimal_img,'Optimal')
# End_Time = time.time()
# print(f"Execution time of global optimal{End_Time - Start_Time} sec")
# cv2.imshow("global optimal",result1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Start_Time = time.time()

# result_loc1 = Local_threshold(optimal_img,100,"Optimal")
# End_Time = time.time()
# print(f"Execution time of local optimal{End_Time - Start_Time} sec")
# cv2.imshow("local optimal",result_loc1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# ### otsu image ###
# otsu_img = cv2.imread("./images/Threshold.png")
# otsu_img = cv2.resize(otsu_img,(300,300))
# otsu_img = cv2.cvtColor(otsu_img, cv2.COLOR_BGR2GRAY)
# Start_Time = time.time()

# result2 = Global_threshold(otsu_img,'Otsu')
# End_Time = time.time()
# print(f"Execution time of global otsu{End_Time - Start_Time} sec")
# cv2.imshow("global otsu",result2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Start_Time = time.time()
# result_loc2 = Local_threshold(otsu_img,100,"Otsu")
# End_Time = time.time()
# print(f"Execution time of local otsu{End_Time - Start_Time} sec")
# cv2.imshow("local otsu",result_loc2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ### spectrat image ###

# spect_img = cv2.imread("./images/Threshold.png")
# spect_img = cv2.resize(spect_img,(300,300))
# spect_img = cv2.cvtColor(spect_img, cv2.COLOR_BGR2GRAY)
# Start_Time = time.time()
# result3 = Global_threshold(spect_img,'spect')
# End_Time = time.time()
# print(f"Execution time of global spectral{End_Time - Start_Time} sec")
# cv2.imshow("global spectral",result3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Start_Time = time.time()
# result_loc3 = Local_threshold(spect_img,100,"spect")
# End_Time = time.time()
# print(f"Execution time of local spectral{End_Time - Start_Time} sec")
# cv2.imshow("local spectral",result_loc3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

