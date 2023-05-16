import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import flow_file_processing as fp
import visualize as vis



def getWarpedSrcImg(srcImg, H, showImages=True):
    """
    Get warped image, for comparison
    :param srcImg: image to be warped back to the destination
    :param H: the H matrix
    :param showImages: display or not
    :return: warped image
    """
    ht, wd, cc = srcImg.shape

    ww = wd
    hh = ht

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    result = cv2.warpPerspective(srcImg, H, (ww, hh))

    if showImages:
        plt.figure(figsize=(10, 4))
        plt.imshow(result)

        plt.show()

    return result


def getIntersection(warp_srcImg, destImg, showImages):
    """
    Obsolete function currently, was to be used to find the square intersection pixels between two circular images.
    Planned to work more on this and improve it to not just square intersection but any shape intersection
    :param warp_srcImg:
    :param destImg:
    :param showImages:
    :return:
    """
    warp_srcImg_gray = cv2.cvtColor(warp_srcImg, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(warp_srcImg_gray)  # Find all non-zero points (text)

    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    cropped_src_img = warp_srcImg[y:y + h, x:x + w]  # Crop the image - note we do this on the original image

    if showImages:
        print("cropped_src_img")
        plt.figure(figsize=(10, 4))
        plt.imshow(cropped_src_img)
        plt.show()

    color = (0, 0, 0)
    masked_dest_img = np.full(destImg.shape, color, dtype=np.uint8)
    
    
    if (coords is None):
      invalid = True
      flag = False
      flag2 = False
      cropped_src_img = 0 
      cropped_dest_img = 0
    else:
      print(coords.shape[0])
      invalid = False
      if(100000<coords.shape[0]< 450000):
        flag = False
        flag2 = False
      else:
        flag = True 
        if coords.shape[0] < 60000:
          flag2 = True
        else:
          flag2 = False
      print(flag)
        
      I, J = np.transpose(coords)
      masked_dest_img[J, I] = destImg[J, I]
      cropped_dest_img = masked_dest_img[y:y + h, x:x + w]

      if showImages:
          print("cropped_dest_img")
          plt.figure(figsize=(10, 4))
          plt.imshow(cropped_dest_img)
          plt.show()
          
    print(invalid)
      
    

    return [cropped_src_img, cropped_dest_img, flag, invalid, flag2]


def getSSIM(src_img, dest_img):
    """
    Get SSIM between two images, note that with a little change to the function ssim , chaning a default parameter
    can get you SSIM map as well. YOu can also use multichannel versions, but this configurations worked best for me.
    :param src_img: src_img
    :param dest_img: dest_img
    :return: SSIM Value.
    """
    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    dest_img = cv2.cvtColor(dest_img, cv2.COLOR_RGB2GRAY)

    src_img = cv2.GaussianBlur(src_img, (3, 3), 1)
    dest_img = cv2.GaussianBlur(dest_img, (3, 3), 1)

    ssim_value = ssim(src_img, dest_img)
    # ssim_value = ssim(src_img, dest_img, multichannel = True)

    return ssim_value


def getHRefToBase(H_frame_distance, transformation):
    """
    Gets transformation to the first frame for a particular distance
    :param H_frame_distance: a distance(e.g 5) x transformation array that contains all pairwise transformations for that length
    :param transformation: Affine or Homography
    :return: returns transformation to base frame.
    """
    frame_distance = len(H_frame_distance)
    H_ref_to_base = np.zeros((frame_distance, 3, 3))

    for i in range(1, frame_distance + 1):
        new_H = np.eye(3);

        if transformation == "Homography":
            for j in range(i):
                new_H = np.matmul(new_H, H_frame_distance[j])
        elif transformation == "Affine":
            for j in range(i):
                new_H = np.matmul(new_H, np.vstack([H_frame_distance[j], [0, 0, 1]]))
        H_ref_to_base[i - 1] = new_H

    return H_ref_to_base

def getSSIMForFrameDistance(window_size, frame_distance, H_array, transformation, squareLength, imgPaths, showImages):
    """
    Get SSIM for a given frame distance away.
    :param window_size: size of stride, normally should be 1.
    :param frame_distance: normally should be 6
    :param H_array: pair wise transformation
    :param transformation: Affine or Homography
    :param squareLength: length of square which we would crop and calculate SSIM for
    :param imgPaths: paths to the image
    :param showImages: display flag
    :return: matrix of SSIM values.
    """
    
    r = (len(imgPaths) - frame_distance) // window_size
    ssimMatrix = np.zeros((r, frame_distance))

    if transformation == "Homography":
        H_frame_distance = np.zeros((frame_distance, 3, 3))
    elif transformation == "Affine":
        H_frame_distance = np.zeros((frame_distance, 2, 3))

    for i in tqdm(range(r)):  # r
        begin = i
        end = i + frame_distance
        img_indexes = np.arange(begin, end + 1)
        #print("img_indexes",img_indexes)
        img_indexes_paths = [imgPaths[i] for i in img_indexes]
        print("img_indexes_paths", img_indexes_paths)

        # fill up H_ref_to_base for this list
        H_frame_distance = H_array[begin:end]

        H_ref_to_base = getHRefToBase(H_frame_distance, transformation)

        for j in range(frame_distance):
            destImgPath = img_indexes_paths[0]  # the previous image
            srcImgPath = img_indexes_paths[j + 1];

            srcImg, destImg = fp.inputAndVisualizeStitchPair(srcImgPath, destImgPath, showImages)

            warp_srcImg = getWarpedSrcImg(srcImg, H_ref_to_base[j], showImages)

            cropped_warp_srcImg = get_square_in_image(warp_srcImg, squareLength, showImages)
            cropped_destImg = get_square_in_image(destImg, squareLength, showImages)

            ssim_value = getSSIM(cropped_warp_srcImg, cropped_destImg)
            ssimMatrix[i, j] = ssim_value
    return ssimMatrix

def findDenseTransformation(srcImg, destImg):
  """
  Get Lucas Kanade pyrammidal dense transformation, between a srcImage and a destimage, using a contrib branch of opencv.
  I do not know why I added a status there, I think it was to help with errors
  but I would check and remove it if it is not needed
  Improvement on this would be to perform the pyramidal transformation myself which would allow me to be able to uses
  our circular mask. But currently it works.
  :param srcImg: srcImg
  :param destImg: destImg
  :return: Dense transformation
  """

  mapper = cv2.reg_MapperGradAffine()
  mapperPyramid = cv2.reg_MapperPyramid(mapper)
  # mapperPyramid.numIterPerScale_ = 3
  # mapperPyramid.numLev_ = 3

  result_pointer = mapperPyramid.calculate(srcImg.astype(float), destImg.astype(float))
  result_array = cv2.reg.MapTypeCaster_toAffine(result_pointer)

  H = np.concatenate([result_array.getLinTr(), result_array.getShift()], axis = 1)
  status = False
  return [H, status]


def get_square_in_image(image, squareLength, showImages):
    """
    Get square images in the middle of warped src image and destination image for performing SSIM
    currently in use, if time allows would change to any shape of intersection via getIntersection()
    :param image: image
    :param squareLength: middle square length to be considered
    :param showImages: display flag
    :return: return square image.
    """
    h_start = int((image.shape[0] - squareLength) / 2)
    w_start = int((image.shape[1] - squareLength) / 2)

    crop_img = image[h_start:h_start + squareLength, w_start:w_start + squareLength]
    if showImages:
        print("cropped Image...............")
        vis.visualizeImg(crop_img)

    return crop_img