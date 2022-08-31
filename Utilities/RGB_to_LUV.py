import numpy as np
import cv2 
import math
import matplotlib.pyplot as plt






def RGB_To_LUV(img):
       

        row,col,_=img.shape
        output = np.zeros(img.shape, dtype=np.uint8)
        
        
        for i in range(row):
            for c in range(col):
              r= img[i,c][0]/255
              g= img[i,c][1]/255
              b= img[i,c][2]/255
              x=  0.412453*r+0.357588*g+0.180423*b
              y=  0.212671*r+0.715169*g+0.072169*b
              z=  0.019334*r+0.119193*g+0.950227*b
              if (y)>0.00856:
                      
                      output[i,c][0]=(round(116*y**(1./3)))
              else:
                      output[i,c][0]=(round(903.3*y))
        
              u_=(4*x/(x+15*y+3*z))
              v_=(9*y/(x+15*y+3*z))
        
        
            output[i,c][1]=(round(13*output[i,c][0]*(u_-0.19793943)))
            output[i,c][2]=(round(13*output[i,c][0]*(v_-0.46831096)))
        
            output[i,c][0]=((255/100)*output[i,c][0])
            output[i,c][1]=((255/354)*output[i,c][1]+134)
            output[i,c][2]=((255/262)*output[i,c][2]+140)

        return output





img1 = cv2.imread("./images/rgb.webp")
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1=np.uint8(img1)
img1=cv2.resize(img1,(255,255))

Output= RGB_To_LUV(img1)

# cv2.imshow('output Image', Output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # From builtin function:

# luv2 = cv2.cvtColor(img1,cv2.COLOR_RGB2Luv)
# cv2.imshow('output Image from builtIn-Fun', luv2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()