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

IMG_SHAPE_STANDARD = (448,448);

def get_mask_im(fullImgPaths, mask_path, crop_top, crop_bottom):
    """
    :param fullImgPaths: Image path of images to be processed, need only the size of one image there
    :param mask_path: path to the mask image to be used
    :param crop_top: the amount of pixels to be removed at the top due to dead pixels
    :param crop_bottom:  the amount of pixels to be removed at the bottom due to dead pixels
    :return: returns mask image
    """
    img_1 = cv2.imread(fullImgPaths[0])
    img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)


    mask_im = Image.open(mask_path)

    mask_im = mask_im.resize((img_1.shape[0], img_1.shape[1]), Image.ANTIALIAS)
    mask_im = np.array(mask_im)
    mask_im = mask_im * np.uint8(255)

    # crop mask
    mask_im[:crop_top] = 0
    mask_im[(mask_im.shape[0] - crop_bottom):] = 0

    mask_im = cv2.resize(mask_im, img_1.shape[1::-1])

    return mask_im


def inputAndVisualizeStitchPair(srcImgPath, destImgPath, showImages=True):
    """
    :param srcImgPath: path to image to be projected back to the destination image
    :param destImgPath: path to the image which is the destination
    :param showImages: bool to show images or not
    :return: returns src image and dest image pair
    """
    srcImg = cv2.imread(srcImgPath)
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)
    srcImg_gray = cv2.cvtColor(srcImg, cv2.COLOR_RGB2GRAY)

    destImg = cv2.imread(destImgPath)
    destImg = cv2.cvtColor(destImg, cv2.COLOR_BGR2RGB)
    destImg_gray = cv2.cvtColor(destImg, cv2.COLOR_RGB2GRAY)

    if showImages:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(10, 4))
        ax1.imshow(destImg, cmap="gray")
        ax1.set_xlabel("dest image", fontsize=14)

        ax2.imshow(srcImg, cmap="gray")
        ax2.set_xlabel("Src image (Image to be transformed)", fontsize=14)

        plt.show()
    return [srcImg, destImg]

def inputAndFormatFlowfile(flowPath):
  '''
  Input: flowPath
  flowpath: path to the .flo file between two images ->str
  Output: flow
  flow: the formated optical_flow -> ndArray of heightxbreathx2 of the image sizes
  '''
  path = Path(flowPath)
  with path.open(mode='r') as flo:
      np_flow = np.fromfile(flo, np.float32)

  with path.open(mode='r') as flo:
    tag = np.fromfile(flo, np.float32, count=1)[0]
    width = np.fromfile(flo, np.int32, count=1)[0]
    height = np.fromfile(flo, np.int32, count=1)[0]

    print('tag', tag, 'width', width, 'height', height)

    nbands = 2
    tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
    flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    return flow

def getCameraPixels(img, mask_im):
    """
    returns I and J which are the pixel positions that are in the mask
    :param img: image in which we want to get this pixels
    :param mask_im: Mask
    :return: returns rows and columns that are in the required round camera as I and J.
    """

    img = cv2.bitwise_and(img, img, mask=mask_im)

    # _,  imgMask = cv2.threshold(img,5,255,cv2.THRESH_BINARY)

    imgMaskGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    nonZero = cv2.findNonZero(imgMaskGray)
    I,J = np.transpose(nonZero).squeeze()

    return [I, J]


def pointCorrespondenceFromOpticalFlow(flow, padding, nonZeroI, nonZeroJ):
    '''
    Input: flow
    flow: the formatted optical flow file -> ndarray length,breadth,2
    padding: padding to add to destImg
    OutPut: [ptsA and ptsB] -> list
    ptsA: point correspondences in the destImg (original) -> ndarray length*breadth,2
    ptsB: point correspondences in the srcImg(new, been transformed) -> ndarray length*breadth,2
    nonZeroI : I rows in the mask
    nonZeroJ: J columns in the mask
    '''
    ptsA = np.zeros(flow.shape)
    ptsB = np.zeros(flow.shape)

    print(flow.shape)

    for i in range(ptsA.shape[0]):
        for j in range(ptsA.shape[1]):
            ptsA[i, j] = np.array([i, j], dtype=np.float)
            ptsB[i, j] = np.array([i, j]) + (np.array(flow[j, i]))

    ptsA = ptsA[nonZeroI, nonZeroJ]
    ptsB = ptsB[nonZeroI, nonZeroJ]

    return [ptsA, ptsB]

def findTransformation(ptsA, ptsB, transformation, threshold=1):
  '''
    ptsA: point correspondences in the destImg (original) -> ndarray length*breadth,2
    ptsB: point correspondences in the srcImg(new, been transformed) -> ndarray length*breadth,2
    threshold: ransac threshold.
    Output:
    H - homography
    status - mask of ransac accepted or rejected
  '''
  if transformation == "Homography":
    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, threshold)
  elif transformation == "Affine":
    (H, status) = cv2.estimateAffine2D(ptsB, ptsA, method = cv2.RANSAC, ransacReprojThreshold = threshold)
  return [H, status]


def pointCorrespondenceFromOpticalFlowSquareCropped(flow):
    """
    Return correspndence when I am using square images and do not care about masks.
    :param flow: flow file
    :return: pts in destination image and the corresponding points in src image.
    """

    ptsA = np.zeros(flow.shape)
    ptsB = np.zeros(flow.shape)

    for i in range(ptsA.shape[0]):
        for j in range(ptsA.shape[1]):
          ptsA[i,j] = np.array([i,j], dtype=np.float)
          ptsB[i,j] = np.array([i,j])  +  (np.array(flow[j, i]))

    ptsA = np.reshape(ptsA, (-1, 2))
    ptsB = np.reshape(ptsB, (-1, 2))
    return [ptsA, ptsB]


