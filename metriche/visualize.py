# visualization module

import cv2
import numpy as np
import matplotlib.pyplot as plt
cv2.ocl.setUseOpenCL(False)
import sys
from PIL import Image
from pathlib import Path
from os import listdir
import os
import shutil

image_width = 448;

def visualizeStitch(srcImg, destImg, H, padding, transformation, mask_im, showImages=True):
    """
    :param srcImg: srcImg
    :param destImg: destImg
    :param H: transformation matrix
    :param padding: padding value if used
    :param transformation: homography or affine
    :param mask_im: mask image
    :param showImages: display flag
    :return: none
    """
    ht, wd, cc = destImg.shape

    ww = wd + (2 * padding)
    hh = ht + (2 * padding)

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    if transformation == "Homography":
        result = cv2.warpPerspective(srcImg, H, (ww, hh))
    elif transformation == "Affine":
        result = cv2.warpAffine(srcImg, H, (ww, hh))

    alpha_s = mask_im / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        result[yy:yy + ht, xx:xx + wd, c] = (alpha_s * destImg[:, :, c] +
                                             alpha_l * result[yy:yy + ht, xx:xx + wd, c])

    print("Visualize stitch")
    if showImages:
        plt.figure(figsize=(10, 4))
        plt.imshow(result)

        plt.show()


def VisualizeOutliers(destImgPath, status, showImages, image_width):
  """
  Outlier visualize function
  :param destImgPath: destination image
  :param status: status of pixel whether outlier or not, from ransac opencv function
  :param showImages: display flag
  :param image_width: width of image
  :return: none
  """
  if showImages:
    ransac_fail = np.where(np.any(status==0, axis=1))[0]
    convert_to_pixel = lambda t: [t//image_width, t%image_width]
    outlier_pixels = np.array([convert_to_pixel(p) for p in ransac_fail])
    # print(outlier_pixels)
    destImg = Image.open(destImgPath)
    fig = plt.figure(figsize=(10,4))
    plt.xlabel("outliers",fontsize=14)
    if (outlier_pixels.size):
      plt.plot(outlier_pixels[:, 1],outlier_pixels[:, 0],'r.')
    plt.imshow(destImg)


def plotPixels(img, I, J):
  """
  plot a single pixel
  :param img:
  :param I:
  :param J:
  :return:
  """
  fig = plt.figure(figsize=(10,4))
  plt.plot(I,J,'r.')
  plt.imshow(img)
  plt.show()


def visualizeImg(img):
  """
  Visualize an image, when I have the image in matrix form
  :param img:
  :return:
  """
  plt.figure(figsize=(10,4))
  plt.imshow(img)
  plt.show()

def ImgFromPath(imgPath, showImages = True):
  """
  Visualize an image when I have the path
  :param imgPath:
  :param showImages:
  :return:
  """
  img = cv2.imread(imgPath)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  if showImages:
    plt.figure(figsize=(10,4))
    plt.imshow(img)
    plt.show()
  return img