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
image = mpimg.imread('./../Car-Tracking-Data/examples/test3.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255
windows = []
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 550], 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[300, 600], 
                    xy_window=(123, 123), xy_overlap=(0.75, 0.75))

hot_windows_pre = []
hot_windows_pre += windows[4:6]
hot_windows_pre += windows[13:16]
hot_windows_pre += windows[25:29]
hot_windows_pre += windows[545:546]
hot_windows_pre += windows[625:626]

# hot_windows = combine_boxes(hot_windows_pre)
hot_windows = hot_windows_pre


hot_windows2 = create_heatmap(hot_windows_pre, image.shape)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)
plt.show()

print(hot_windows2.shape)
blobs_doh = blob_doh(hot_windows2, max_sigma=200, threshold=.08)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
blobs = blobs_doh
color = 'red'

ax.imshow(window_img, interpolation='nearest')
for blob in blobs:
    y, x, r = blob
    c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
    ax.add_patch(c)

plt.show()

windows = combine_boxes(hot_windows_pre, image.shape)

window_img = draw_boxes(draw_image, windows, color=(0,0,255), thick = 6)
plt.imshow(window_img)
plt.show()

