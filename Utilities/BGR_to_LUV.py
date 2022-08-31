import numpy as np
import cv2 


def BGR_To_LUV(img):
       

        height,width,_=img.shape
        output = np.zeros(img.shape)
                
        for i in range(height):
            for c in range(width):

                b = img[i,c][0]/255.0
                g = img[i,c][1]/255.0
                r = img[i,c][2]/255.0


                x =  (0.412453*r) + (0.35758*g) + (0.180423*b)
                y =  (0.212671*r) + (0.71516*g) + (0.072169*b)
                z =  (0.019334*r) + (0.119193*g) + (0.950227*b)

                if (y) > 0.008856:
                        output[i,c][0] = (116.0 * (y**(1.0/3))) - 16

                elif (y) <= 0.008856:
                        output[i,c][0] = (903.3*y)
                        print("<=")

                u_dash= 4.0 * x / (x + (15.0*y) + (3.0*z))
                v_dash= 9.0 * y / (x + (15.0*y) + (3.0*z))

                output[i,c][1] = 13 * output[i,c][0] * (u_dash - 0.19793943)
                output[i,c][2] = 13 * output[i,c][0] * (v_dash - 0.46831096)

                output[i,c][0]=((255/100)*output[i,c][0])
                output[i,c][1]=((255/354)*output[i,c][1]+134)
                output[i,c][2]=((255/262)*output[i,c][2]+140)

                # output[i,c][0] = round(output[i,c][0] / 100.0 *255)
                # output[i,c][1] = round((output[i,c][1] + 134.0) / 354 * 255)
                # output[i,c][2] = round((output[i,c][2] + 140.0) / 262 * 255)

        return output.astype(np.uint8)





# img1 = cv2.imread("fruit.png")
# Output = cv2.cvtColor(img1, cv2.COLOR_BGR2LUV)

# cv2.imshow('output Image', Output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Output= BGR_To_LUV(img1)

# cv2.imshow('output Image', Output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # From builtin function:

# luv2 = cv2.cvtwidthor(img1,cv2.widthOR_RGB2Luv)
# cv2.imshow('output Image from builtIn-Fun', luv2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()