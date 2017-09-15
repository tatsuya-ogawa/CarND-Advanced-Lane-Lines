import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

from functions import *

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist = calibrate(images)


def process_image(src_img, file_name=None, plot=False):
    undistorted_img = cv2.undistort(src_img, mtx, dist, None, mtx)

    # apply sobel filters
    img_hls_sbs_sobel = hls_abs_sobel(undistorted_img, s_thresh=(170, 255), sx_thresh=(20, 100))
    img_dir_sobel = dir_threshold_sobel(undistorted_img, sobel_kernel=15, thresh=(0.7, 1.3))
    img_mag_sobel = mag_thresh_sobel(undistorted_img, sobel_kernel=3, mag_thresh=(30, 100))
    combined = np.zeros_like(img_hls_sbs_sobel)

    combined[(img_hls_sbs_sobel == 1) | ((img_dir_sobel == 1) & (img_mag_sobel == 1))] = 1

    # warp image
    h, w = undistorted_img.shape[:2]
    src = np.float32([(575, 464),
                      (707, 464),
                      (258, 682),
                      (1049, 682)])
    dst = np.float32([(450, 0),
                      (w - 450, 0),
                      (450, h),
                      (w - 450, h)])

    warped, Minv = warp(combined, src, dst)

    windows, left_lane_inds, right_lane_inds = sliding_window(warped)

    windowed = None
    if plot or file_name is not None:
        windowed = draw_window(warped, windows, left_lane_inds, right_lane_inds)

    lane = draw_lane(undistorted_img, warped, left_lane_inds, right_lane_inds, Minv)

    # Display processed images if plot is True
    if plot:
        f, axs = plt.subplots(3, 4, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        axs = axs.ravel()
        index = 0

        axs[index].imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
        axs[index].set_title('Original Image', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        axs[index].set_title('Undistorted Image', fontsize=20)
        axs[index].axis('off')
        index = index + 1
        axs[index].axis('off')
        index = index + 1
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(img_mag_sobel, cmap='gray')
        axs[index].set_title('Sobel Magnitude', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(img_dir_sobel, cmap='gray')
        axs[index].set_title('Sobel Dir', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(img_hls_sbs_sobel, cmap='gray')
        axs[index].set_title('Sobel Hls', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(combined, cmap='gray')
        axs[index].set_title('Pipelined Image', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(warped, cmap='gray')
        axs[index].set_title('Warped Image', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(windowed)
        axs[index].set_title('Sliding window', fontsize=20)
        axs[index].axis('off')
        index = index + 1

        axs[index].imshow(cv2.cvtColor(lane, cv2.COLOR_BGR2RGB))
        axs[index].set_title('Lane', fontsize=20)
        axs[index].axis('off')
        index = index + 1
        axs[index].axis('off')

        plt.show()

    # Output processed images if file_name is not None
    if file_name is not None:
        from os.path import basename
        file_name = basename(file_name)
        cv2.imwrite('./output_images/org_' + file_name, src_img)
        cv2.imwrite('./output_images/undist_' + file_name, undistorted_img)
        cv2.imwrite('./output_images/mag_sobel_' + file_name, img_mag_sobel * 255)
        cv2.imwrite('./output_images/dir_sobel_' + file_name, img_dir_sobel * 255)
        cv2.imwrite('./output_images/hls_sobel_' + file_name, img_hls_sbs_sobel * 255)
        cv2.imwrite('./output_images/combined_' + file_name, combined * 255)
        cv2.imwrite('./output_images/warped_' + file_name, warped * 255)
        cv2.imwrite('./output_images/windowed_' + file_name, windowed)
        cv2.imwrite('./output_images/lane_' + file_name, lane)
    return lane


images = glob.glob('./test_images/test*.jpg')

# for fname in images:
#     img = cv2.imread(fname)
#     process_image(img, file_name=fname, plot=True)

video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')  # .subclip(22,26)
processed_video = video_input1.fl_image(process_image)
processed_video.write_videofile(video_output1, audio=False)
