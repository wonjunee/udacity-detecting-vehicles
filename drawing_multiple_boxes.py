import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import copy
from lesson_functions import *
from scipy import ndimage as ndi
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, blob_doh, peak_local_max
from skimage.morphology import watershed
from sklearn.model_selection import train_test_split


# Drawing random boxes on an image
image = mpimg.imread('./../Car-Tracking-Data/examples/test5.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255
windows = []
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))

windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(144, 144), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[430, 550], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[460, 580], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))

# Creating random windows
# hot_windows_pre = []
# hot_windows_pre += windows[4:6]
# hot_windows_pre += windows[13:16]
# hot_windows_pre += windows[25:29]
# hot_windows_pre += windows[545:546]
# hot_windows_pre += windows[625:626]
# hot_windows = hot_windows_pre

hot_windows = windows
# Draw an image with boxes
window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)
plt.title('draw boxes')
plt.show()

# # Draw an image with blobs
# hot_windows2 = create_heatmap(hot_windows, image.shape)
# blobs_doh = blob_doh(hot_windows2, max_sigma=200, threshold=.1)
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal')
# blobs = blobs_doh
# color = 'red'
# ax.imshow(window_img, interpolation='nearest')
# for blob in blobs:
#     y, x, r = blob
#     c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#     ax.add_patch(c)
# plt.title('draw circles')
# plt.show()

# # Draw heatmap with circles
# # Generate an initial image with two overlapping circles
# image = np.zeros(image.shape[:2])
# y, x = np.indices(image.shape[:2])
# # filtered_windows = blobs_to_windows(blobs, hot_windows)
# # filtered_windows = create_heatmap(filtered_windows, image.shape)
# # image = np.logical_or(image, filtered_windows)
# hot_windows2[hot_windows2>0] = 1
# image = hot_windows2

# # Now we want to separate the two objects in image
# # Generate the markers as local maxima of the distance to the background
# distance = ndi.distance_transform_edt(image)
# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#                             labels=image)
# markers = ndi.label(local_maxi)[0]
# labels = watershed(-distance, markers, mask=image)

# fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
# ax0, ax1, ax2 = axes

# ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# ax0.set_title('Overlapping objects')
# ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
# ax1.set_title('Distances')
# ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
# ax2.set_title('Separated objects')
# for ax in axes:
#     ax.axis('off')

# fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.9, bottom=0, left=0,
#                     right=1)
# plt.show()

# windows = []
# for blob in blobs:
#     y, x, r = blob
#     num = labels[int(y), int(x)]
#     im = np.zeros(image.shape[:2])
#     im[labels == num] = 1
#     print(np.argwhere(im))
#     starty = (np.min(np.argwhere(im)[:,0]))
#     endy = (np.max(np.argwhere(im)[:,0]))
#     startx = (np.min(np.argwhere(im)[:,1]))
#     endx = (np.max(np.argwhere(im)[:,1]))
#     windows.append(((int(startx), int(starty)), (int(endx), int(endy))))

# # Draw an image with boxes
# window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)                    
# plt.imshow(window_img)
# plt.title('applying watershed')
# plt.show()

# # Repeat the entire procedure with a single function
# windows = combine_boxes(hot_windows, image.shape, max_sigma=200, threshold=0.1)
# window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)                    
# plt.imshow(window_img)
# plt.title('single function')
# plt.show()

