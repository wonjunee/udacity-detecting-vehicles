import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from lesson_functions import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Reload the data
pickle_file = './../Car-Tracking-Data/svc.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    svc = pickle_data['svc']
    X_scaler = pickle_data['X_scaler']
    del pickle_data  # Free up memory

print('Model loaded.')

image = mpimg.imread('./../Car-Tracking-Data/examples/test3.jpg')
image = image.astype(np.float32)/255
windows = []
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 550], 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[300, 600], 
                    xy_window=(192, 192), xy_overlap=(0.5, 0.5))

def process_image(image):
    # Save a raw image
    draw_image = np.copy(image)
    # Normalize image
    image = image.astype(np.float32)/255
    # Apply pipeline to the image to create black and white image
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    # Combine overlapping windows
    a = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=4)  
    # hot_windows = combine_boxes(hot_windows, image.shape, max_sigma=max_sigma, threshold=0.15)
    # Average over windows with previous windows
    # Combine overlapping windows
    hot_windows, labels = combine_boxes(hot_windows, image.shape)
    if len(Window.windows1) == 0:
        Window.windows1 = hot_windows
        Window.windows2 = hot_windows
        Window.windows3 = hot_windows
    # Average windows over windows from previous frames
    results = average_boxes(hot_windows, 
        Window.windows1, Window.windows2, Window.windows3,
        image.shape)
    # Reassign window values in a class
    Window.windows3 = copy.copy(Window.windows2)
    Window.windows2 = copy.copy(Window.windows1)
    Window.windows1 = results
    # Return the original image with boxes    
    return draw_boxes(a, hot_windows, color=(0, 0, 255), thick=6), labels

Window = Window()
for i in range(1,6):
    image = mpimg.imread('./../Car-Tracking-Data/examples/test{}.jpg'.format(i))
    
    Window.windows1 = []

    window_img, labels = process_image(image)
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax0, ax1 = axes
    ax0.imshow(window_img)
    ax1.imshow(labels)
    plt.show()

    # window_img = process_image(image)
    # plt.imshow(window_img)
    # plt.show()
