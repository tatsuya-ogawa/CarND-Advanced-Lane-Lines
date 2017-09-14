import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from functions import *

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist = calibrate(images)

images = glob.glob('./test_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img_hls_sbs_sobel = hls_abs_sobel(img, s_thresh=(170, 255), sx_thresh=(20, 100))
    img_dir_sobel = dir_threshold_sobel(img, sobel_kernel=15, thresh=(0.7, 1.3))
    img_mag_sobel = mag_thresh_sobel(img, sobel_kernel=3, mag_thresh=(30, 100))
    combined = np.zeros_like(img_hls_sbs_sobel)

    combined[(img_hls_sbs_sobel == 1) | ((img_dir_sobel == 1) & (img_mag_sobel == 1))] = 1
    warped, Minv = warp(combined)
    windowed, leftx, lefty, rightx, righty = sliding_window(warped)
    # wid2 = sliding_window2(warped)
    lane = draw_lane(img, warped, leftx, lefty, rightx, righty, Minv)

    f, axs = plt.subplots(2, 4, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.05)
    axs = axs.ravel()
    axs[0].imshow(img)
    axs[0].set_title('Undistorted Image', fontsize=30)
    axs[1].imshow(img_mag_sobel, cmap='gray')
    axs[1].set_title('Sobel Magnitude', fontsize=30)
    axs[2].imshow(img_dir_sobel, cmap='gray')
    axs[2].set_title('Sobel Dir', fontsize=30)
    axs[3].imshow(img_hls_sbs_sobel, cmap='gray')
    axs[3].set_title('Sobel Hls', fontsize=30)
    axs[4].imshow(combined, cmap='gray')
    axs[4].set_title('Sobel Combine', fontsize=30)
    axs[5].imshow(warped, cmap='gray')
    axs[5].set_title('Warped', fontsize=30)
    axs[6].imshow(windowed, cmap='gray')
    axs[6].set_title('Sliding window1', fontsize=30)
    axs[7].imshow(lane)
    axs[7].set_title('Lane', fontsize=30)

    plt.show()
