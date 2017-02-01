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
pickle_file = './../../Car-Tracking-Data/svc.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    svc = pickle_data['svc']
    X_scaler = pickle_data['X_scaler']
    parameters = pickle_data['parameters']
    del pickle_data  # Free up memory


color_space = parameters['color_space']
orient = parameters['orient']
pix_per_cell = parameters['pix_per_cell']
cell_per_block = parameters['cell_per_block']
hog_channel = parameters['hog_channel']
spatial_size = parameters['spatial_size']
hist_bins = parameters['hist_bins']
spatial_feat = parameters['spatial_feat']
hist_feat = parameters['hist_feat']
hog_feat = parameters['hog_feat']

print('Model and parameters loaded.')

image = mpimg.imread('./../../Car-Tracking-Data/examples/test3.jpg')

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))

windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(144, 144), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[430, 550], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[460, 580], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))

# This function will draw boxes on the image
# Input: Original image
# Output: Original image with boxes
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
    # # Combine overlapping windows
    draw_image = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)
    combined_windows = combine_boxes(hot_windows, image.shape)

    # Average windows over windows from previous frames
    results, Window.probability = average_boxes(hot_windows, 
                                    Window.probability,
                                    image.shape)

    # Return an image with boxes drawn
    return draw_boxes(draw_image, results, color=(0, 0, 255), thick=3), hot_windows

Window = Window()

for i in range(1,2):
    image = mpimg.imread('./../../Car-Tracking-Data/examples/test{}.jpg'.format(i))
    
    Window.probability = []

    window_img, hot_windows = process_image(image)
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax0, ax1 = axes
    ax0.imshow(window_img)
    ax0.set_title("Image With Windows")
    labels = create_heatmap(hot_windows, image.shape)
    ax1.imshow(labels, cmap='hot')
    ax1.set_title("Heatmap")
    plt.show()
