| Member Name    | Sec     | BN  
| ------------- | ------------- | --------    |
| Ahmed khaled hilal | 1         |   3   |
| Dalia lotfy Abdulhay | 1        | 30   |
| Radwa Saeed Mohammady | 1        | 33   |
| Meeran Ahmed Mostafa | 2        | 34  |
| Yousef Samir | 2       | 49   |
| **Group NO.**    |  | 8   |
**Contact mail** : meeranahmed147@gmail.com
------

# Results :

## Thresholding:

1- **Optimal Threshold**


- Original image

  ![](images/threshold.png)

- After Applying Global optimal threshold

  ![](Output/globalOptimal_out.png)
  

- After Applying local optimal threshold

  ![](Output/localOptimal_out.png)

2- **Otsu Threshold**


- Original image

  ![](images/threshold.png)

- After Applying Global Otsu threshold

  ![](Output/globalOtsu_out.png)
  

- After Applying local Otsu threshold

  ![](Output/localOtsu_out.png)

3- **Spectral Threshold**


- Original image

  ![](images/threshold.png)

- After Applying Global Spectral threshold

  ![](Output/globalSpectral_out.png)
  

- After Applying local Spectral threshold

  ![](Output/localSpectral_out.png)
-------

## Conversion from RGB to LUV :

- Original image

  ![](images/rgb.png)
  

- After Applying luv conversion

  ![](Output/LUV_out.png)


## Segmentation:
1- **K-Mean**
- Original image

  ![](images/kmean+meanShift.png)


- After Segmentation

  ![](Output/Kmeans_out.png)


2- **Mean Shift**
- Original image

  ![](images/kmean+meanShift.png)

- After Segmentation

  ![](Output/meanShift_out.png)


3- **Agglomerative method**
- Original image

  ![](images/AGG_img.png)

- After Segmentation

  ![](Output/AGGlomerative_result.png)


4- **Region growing**
- Original image

  ![](images/REG_img.png)

- After Segmentation

  ![](Output/reg_out.png)

# Ui Screenshots :
![](screenshots/1.png)
![](screenshots/3.png)
![](screenshots/4.png)


# Computation Time :


### Thresholding
----

- Optimal

![](Computation_time/optimal.png)

- Otsu 

![](Computation_time/otsu.png)


- Spectral 
![](Computation_time/spectral1.png)
![](Computation_time/spectral2.png)

-------
### Segmentation
-----
- K- Mean

![](Computation_time/kmean1.jpeg)
![](Computation_time/kmean2.jpeg)



- MeanShift

![](Computation_time/meanshift.png)

- Agglomerative 

![](Computation_time/agg.png)

- Region growing

![](Computation_time/regionGrowing.png)


### Note : Agglomerative method takes long time to compute, so we use small image as input (that is the reason of appearing pixels in image. )
