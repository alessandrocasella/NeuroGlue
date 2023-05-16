import cv2
import numpy as np
from random import randrange
import random

file_name = r"C:\Users\annad\PycharmProjects\THESIS\final_patches_video1\patch0000.jpg"
image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

f = random.randint(200,800) #400
rotZval = random.randint(0,100)
distZval = random.randint(100,500) #zoom

warped = np.zeros_like(image)
h, w = image.shape[:2]

rotX = 0
rotY = 0
rotZ = (rotZval - 60)*np.pi/120
distX = 0
distY = 0
distZ = distZval - 400

# Camera intrinsic matrix
K = np.array([[f, 0, w/2, 0],
            [0, f, h/2, 0],
            [0, 0,   1, 0]])

# K inverse
Kinv = np.zeros((4,3))
Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*f
Kinv[-1,:] = [0, 0, 1]

# Rotation matrices around the X,Y,Z axis
RX = np.array([[1,           0,            0, 0],
            [0,np.cos(rotX),-np.sin(rotX), 0],
            [0,np.sin(rotX),np.cos(rotX) , 0],
            [0,           0,            0, 1]])

RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
            [            0, 1,            0, 0],
            [ -np.sin(rotY), 0, np.cos(rotY), 0],
            [            0, 0,            0, 1]])

RZ = np.array([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
            [ np.sin(rotZ), np.cos(rotZ), 0, 0],
            [            0,            0, 1, 0],
            [            0,            0, 0, 1]])

# Composed rotation matrix with (RX,RY,RZ)
R = np.linalg.multi_dot([ RX , RY , RZ ])

# Translation matrix
T = np.array([[1,0,0,distX],
            [0,1,0,distY],
            [0,0,1,distZ],
            [0,0,0,1]])

# Overall homography matrix
H = np.linalg.multi_dot([K, R, T, Kinv])

# Apply matrix transformation
cv2.warpPerspective(image, H, (w, h), warped, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

# Show the image
cv2.imshow('source', image)
cv2.imshow('warped', warped)

cv2.waitKey()





