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
# Import a library needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# Reload the data
pickle_file = './../Car-Tracking-Data/features.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    X_test = pickle_data['test_dataset']
    y_test = pickle_data['test_labels']
    X_scaler = pickle_data['X_scaler']
    parameters = pickle_data['parameters']
    del pickle_data  # Free up memory

print('Data and modules loaded.')
print("train_features size:", X_train.shape)
print("train_labels size:", y_train.shape)
print("test_features size:", X_test.shape)
print("test_labels size:", y_test.shape)
for k in parameters:
    print(k, ":", parameters[k])

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

print('\nUsing:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')

# Use a linear SVC 
svc = LinearSVC(max_iter=10000)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

image = mpimg.imread('./../Car-Tracking-Data/examples/test3.jpg')

windows = []
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[380, 520], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350, 550], 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[300, 600], 
                    xy_window=(144, 144), xy_overlap=(0.5, 0.5))
# windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[250, 650], 
#                     xy_window=(192, 192), xy_overlap=(0.5, 0.5))

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
    # Combine overlapping windows
    hot_windows, _ = combine_boxes(hot_windows, image.shape)
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
    
    # Return an image with boxes drawn
    return draw_boxes(draw_image, results, color=(0, 0, 255), thick=6)  

Window = Window()
# Draw boxes on a video stream
white_output = './../Car-Tracking-Data/white.mp4' # New video
clip1 = VideoFileClip('./../Car-Tracking-Data/project_video_shortened3.mp4') # Original video
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

