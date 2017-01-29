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
print("parameters:")
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

max_sigma = 150

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
    if len(Window.current_windows) == 0:
        pass
    elif len(Window.previous_windows) == 0:
        hot_windows += Window.current_windows
    else:
        hot_windows += Window.current_windows
        hot_windows += Window.previous_windows
    hot_windows, labels = combine_boxes(hot_windows, image.shape, max_sigma=max_sigma, threshold=0.3)
    Window.previous_windows = Window.current_windows
    if len(hot_windows) > 0:
        Window.current_windows = hot_windows
    # Return the original image with boxes    
    return draw_boxes(a, hot_windows, color=(0, 0, 255), thick=6), labels

Window = Window()
for i in range(1,6):
    image = mpimg.imread('./../Car-Tracking-Data/examples/test{}.jpg'.format(i))
    
    Window.current_windows = []
    Window.previous_windows = []    

    window_img, labels = process_image(image)
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax0, ax1 = axes
    ax0.imshow(window_img)
    ax1.imshow(labels)
    plt.show()

    # window_img = process_image(image)
    # plt.imshow(window_img)
    # plt.show()
