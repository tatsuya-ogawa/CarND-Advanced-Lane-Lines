import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')


def calibrate(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.

    # Step through the list and search for chessboard corners
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    return mtx, dist


mtx, dist = calibrate(images)


def dummy():
    for fname in images:
        img = cv2.imread(fname)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img', img)
        dist_img = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imshow('dist', dist_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


images = glob.glob('./test_images/straight_lines*.jpg')


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # return color_binary

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


for fname in images:
    img = cv2.imread(fname)
    grad_binary = pipeline(img)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    shape = img.shape
    src = np.float32([
        [1170 - 450, shape[0] - 280],
        [1170, shape[0]],
        [150, shape[0]],
        [150 + 410, shape[0] - 280],
    ])
    dst = np.float32([
        [1170, shape[0] - 280],
        [1170, shape[0]],
        [150, shape[0]],
        [150, shape[0] - 280],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped)
    ax2.set_title('Warped Image')
    # ax2.imshow(grad_binary, cmap='gray')
    # ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
