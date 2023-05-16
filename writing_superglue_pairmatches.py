import cv2
import numpy as np
import time

def calculate_homography(in_pts, out_pts):
    """
    in_pts = H*out_pts
    :param in_pts: correspond to src
    :param out_pts:
    :return:
    """

    if isinstance(in_pts, list):
        in_pts = np.array(in_pts)

    if isinstance(out_pts, list):
        out_pts = np.array(out_pts)

    mat_A, mat_b = build_sys_equations(in_pts, out_pts)

    H = np.matmul(np.linalg.pinv(mat_A), mat_b)

    # print(mat_b)
    #
    # print(np.matmul(mat_A, H))

    H = np.reshape(np.hstack((H,1)), (3,3))

    return H

def build_sys_equations(in_pts, out_pts):
    """
    :param in_pts: nparray [[x1, y1], [x2, y2], ...]
    :param out_pts: nparray [[x1, y1], [x2, y2], ...]
    :param include_perp_bisector:
    :return:
    """

    mat_A = np.zeros((np.size(in_pts), 8))
    mat_b = in_pts.ravel()

    i = 0
    for x, y in out_pts:
        # x row
        mat_A[i][0:3] = [x, y, 1]
        mat_A[i][-2:] = [-x*mat_b[i], -y*mat_b[i]]

        # y row
        mat_A[i+1][-5:] = [x, y, 1, -x*mat_b[i+1], -y*mat_b[i+1]]

        # row counter
        i = i+2

    return mat_A, mat_b

def compute_homography(p1, p2):
    A = []
    if len(p1)<len(p2):
        min_len = len(p1)
    else:
        min_len = len(p2)
    for i in range(0, min_len):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


def findHomography(image_1_kp, image_2_kp, matches):
    """gets two matches and calculate the homography between two images
    Args:
        image_1_kp (np array): keypoints of image 1
        image_2_kp (np_array): keypoints of image 2
        matches (np array): matches between keypoints in image 1 and image 2
    Returns:
        np arrat of shape [3,3]: Homography matrix
    """
    # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    for i in range(0, len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(
        image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography


def process_frame(frame_cur, frame_prev, H_old, img_out, num_frame):
    """gets an image and processes that image for mosaicing
    Args:
        frame_cur (np array): input of current frame for the mosaicing
    """
    #CROPPING THE IMAGE

    y = 100
    x = 100
    h = 450
    w = 605
    frame_cur = frame_cur[y:y + h, x:x + w]
    if(num_frame == 0):
        frame_prev = frame_prev[y:y + h, x:x + w]

    str = r"/home/amdeluca/ResultsSG_withSP_blur2/frame{:04d}_frame{:04d}_matches.npz".format((num_frame),(num_frame+1))
    file = np.load(str)

    kp_prev = file['keypoints0']
    kp_cur = file['keypoints1']
    matches = file['matches']

    print(kp_cur.shape)
    print(kp_prev.shape)
    print(len(matches))

    valid = matches > 1
    kp_prev = kp_prev[valid]
    kp_cur = kp_cur[matches[valid]]

    print(kp_cur.shape)
    print(kp_prev.shape)

    # if len(matches) < 4:
    #     return

    # Sort the matches  in the order of their distance.
    #matches = sorted(matches, key=lambda x: x.distance)
    # get the maximum of 20  best matches
    #matches = matches[:min(len(matches), 20)]
    # Draw first 10 matches.

    #match_img = cv2.drawMatches(frame_cur, kp_cur, frame_prev, kp_prev, matches, None,
                                #flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow('matches', match_img)

    #H = findHomography(kp_cur, kp_prev, matches)

    #kp_cur = eye(10000, 2, cv2.CV_32S)

    # kp_cur = cv2.Mat(10000, 2, cv2.CV_32FC2)
    # kp_prev = cv2.Mat(10000, 2, cv2.CV_32FC2)
    #
    #kp_cur = kp_cur.reshape((len(kp_cur),1,2))
    #kp_prev = kp_prev.reshape((len(kp_prev),1,2))

    #h_cur, w_cur = kp_cur.shape
    # vis2 = np.array((len(kp_cur),2,3), cv2.CV_32FC3)
    # vis2 = cv2.CreateMat(h_cur, w_cur, cv2.CV_32FC3)
    # vis0 = cv2.fromarray(kp_cur)
    # kp_cur = cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)

    #kp_cur = np.expand_dims(kp_cur, 2)
    #kp_prev = np.expand_dims(kp_prev, 2)

    H, mask = cv2.findHomography(kp_cur, kp_prev, cv2.RANSAC, ransacReprojThreshold=3.0)

    #H = calculate_homography(image_1_points, image_2_points)
    #H = calculate_homography(kp_prev, kp_cur)

    print(num_frame)

    # mat = np.matrix(H)
    #
    # name = r'C:\Users\annad\PycharmProjects\THESIS\Matrix_homography\superglue\mat{:04d}.txt'.format(num_frame)
    # f = open(name, "w+")
    # # with open(name) as f:
    # for line in mat:
    #     np.savetxt(f, line, fmt='%.5f')
    #
    # f.close()

    H = np.matmul(H_old, H)

    mosaic = warp(frame_cur, H, img_out, num_frame)

    return mosaic, H

def warp(frame_cur, H, img_out, num_frame):
    """ warps the current frame based of calculated homography H
    Args:
        frame_cur (np array): current frame
        H (np array of shape [3,3]): homogaphy matrix
    Returns:
        np array: image output of mosaicing
    """

    warped_img = cv2.warpPerspective(
        frame_cur, H, (img_out.shape[1], img_out.shape[0]), flags=cv2.INTER_LINEAR)

    #name = './Frames_canvas/frameCanvas{:05d}.jpg'.format(num_frame)  # to save the frame in jpg format
    #print('Creating...' + name)
    #cv2.imwrite(name, warped_img)

    # name = r'C:\Users\annad\PycharmProjects\THESIS\FramesSG_canvas\frame{:05d}.jpg'.format(num_frame)
    # print('Creating...' + name)
    # cv2.imwrite(name, warped_img)

    transformed_corners = get_transformed_corners(frame_cur, H)
    warped_img = draw_border(warped_img, transformed_corners)

    img_out[warped_img > 0] = warped_img[warped_img > 0]

    output_temp = np.copy(img_out)
    output_temp = draw_border(output_temp, transformed_corners, color=(0, 0, 255))

    cv2.imshow('output', output_temp)
    out.write(output_temp)


    return img_out


def get_transformed_corners(frame_cur, H):
    """finds the corner of the current frame after warp
    Args:
        frame_cur (np array): current frame
        H (np array of shape [3,3]): Homography matrix
    Returns:
        [np array]: a list of 4 corner points after warping
    """
    corner_0 = np.array([0, 0])
    corner_1 = np.array([frame_cur.shape[1], 0])
    corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
    corner_3 = np.array([0, frame_cur.shape[0]])

    corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    transformed_corners = np.array(transformed_corners, dtype=np.int32)
    #mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
    #cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
    #cv2.imshow('mask', mask)

    return transformed_corners

def draw_border(image, corners, color=(0, 0, 0)):
    """This functions draw rectancle border
    Args:
        image ([type]): current mosaiced output
        corners (np array): list of corner points
        color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).
    Returns:
        np array: the output image with border
    """
    for i in range(corners.shape[1] - 1, -1, -1):
        cv2.line(image, tuple(corners[0, i, :]), tuple(
            corners[0, i - 1, :]), thickness=5, color=color)
    return image

def main():

    global out
    global frame_cur
    global frame_prev
    global H_old
    global kp_prev
    global des_prev
    global descriptor
    global matches
    global mosaic

    tic = time.perf_counter()

    wid_out = 2500
    hei_out = 2500

    num_frame = 0

    img_out = np.ones((wid_out, hei_out, 3), np.uint8)  # creation of the black canvas
    #img_out.fill(255)

    video_path = r"/home/amdeluca/Copia di __2021-02-15_10-34-19_V-00.00.05.040-00.01.11.802.mpg"
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_prova_tempi.avi', fourcc, 25.0, (wid_out, hei_out), True)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', (1920, 1080)) #screen resolution = 1920x1080

    is_first_frame = True
    cap.read()
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            if is_first_frame:
                continue
            break

        if is_first_frame:

            first_image = frame_cur
            w_offset = int(img_out.shape[0] / 2 - first_image.shape[0] / 2)
            h_offset = int(img_out.shape[1] / 2 - first_image.shape[1] / 2)

            frame_prev = first_image
            # y = 100
            # x = 100
            # h = 450
            # w = 605
            # first_image = first_image[y:y + h, x:x + w]

            # img_out[w_offset:w_offset + first_image.shape[0],
            # h_offset:h_offset + first_image.shape[1], :] = first_image

            H_old = np.eye(3)
            H_old[0, 2] = h_offset
            H_old[1, 2] = w_offset

            # mat = np.matrix(H_old)
            #
            # name = r'C:\Users\annad\PycharmProjects\THESIS\Matrix_homography\superglue\mat0000.txt'
            # f = open(name, "w+")
            # # with open(name) as f:
            # for line in mat:
            #     np.savetxt(f, line, fmt='%.5f')
            #
            # f.close()
            #
            is_first_frame = False
            continue

        # process each frame
        mosaic, H = process_frame(frame_cur, frame_prev, H_old, img_out, num_frame)
        num_frame = num_frame +1
        frame_prev = frame_cur
        H_old = H
        name = r'C:\Users\annad\PycharmProjects\THESIS\Networks\MosaicSG\mosaic{:04d}.jpg'.format(num_frame)
        #cv2.imwrite(name, mosaic)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    #cv2.imwrite('mosaic.jpg', mosaic)

    out.release()

    toc = time.perf_counter()

    print(f"The execution time is {toc - tic:0.4f} seconds")

    diff = time.strftime('%H:%M:%S', time.gmtime(toc - tic))
    print(f"The execution time is {diff} minutes")

if __name__ == "__main__":
    main()
