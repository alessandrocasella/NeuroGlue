import numpy as np
import cv2
import os
import ssim_processing as sp
import pandas as pd

videoToTest = "/home/amdeluca/metriche/data_video3"

print(videoToTest)
index = 0
y = 0

score_matrix = np.zeros((3536, 5)) #5
score_matrix_crop = np.zeros((3536, 5)) #5

experiments = ["/brisk", "/orb", "/sift", "/SP", "/SG_trained"]

for experiment in experiments:
    print(experiment)
  

    for index in range(0, 3536):

        path1 = os.path.join(videoToTest + experiment + "/images/" + "frame{:04d}.jpg".format(index))
        path2 = os.path.join(videoToTest + experiment + "/images/" + "frame{:04d}.jpg".format(index + 5))

        frame1 = cv2.imread(path1)
        frame2 = cv2.imread(path2)
        
        crop1, crop2, flag, invalid, flag2 = sp.getIntersection(frame1, frame2, False)
        
        if invalid == True:
            ssim_value_crop = 0
            print('invalid value')
        else:
            ssim_value_crop = sp.getSSIM(crop1, crop2)
            
            if flag == True:
              if experiment == "/brisk" or experiment == "/orb":
                  if flag2 == False:
                      ssim_value_crop = ssim_value_crop/2
                      print('Abnormal type 1 occur')
                  else:
                      ssim_value_crop = ssim_value_crop/3
                      print('Abnormal type 2 occur')
              if experiment == "/sift":
                  ssim_value_crop = ssim_value_crop/2
              
                  
            
            print(experiment, index, index +5, ssim_value_crop)
        
            
        score_matrix_crop[index, y] = ssim_value_crop
            
    
            #ssim_value = sp.getSSIM(frame1, frame2)
            #print(experiment, index, index +5, 'normal', ssim_value)
            
        #score_matrix[index, y] = ssim_value

    print(experiment + ' done')
    media = np.mean(score_matrix_crop[:, y])
        
    y = y + 1

print(score_matrix.shape)
print('Finished')

#df = pd.DataFrame(score_matrix)
#path = os.path.join(videoToTest + "/ssim_matrix_sp.xlsx")
#df.to_excel(excel_writer=path)

df_crop = pd.DataFrame(score_matrix_crop)
path = os.path.join(videoToTest + "/ssim_matrix_sp.xlsx")
df_crop.to_excel(excel_writer=path)


