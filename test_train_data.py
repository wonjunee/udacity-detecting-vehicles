import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import glob
import time
import pickle
from lesson_functions import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

save_features = False
save_model = True
# Define the version from the command line
parser = argparse.ArgumentParser(description='Create Pickles')
parser.add_argument('features', type=str, help='0 or 1')
parser.add_argument('model', type=str, help='0 or 1')
args = parser.parse_args()
save_features = int(args.features)
save_model = int(args.model)

if save_features:
    # Read in cars and notcars
    cars = []
    notcars = []

    cars_images = glob.glob('./../../Car-Tracking-Data/vehicles/*')
    for folder in cars_images:
        cars += glob.glob('{}/*.png'.format(folder))

    notcars_images = glob.glob('./../../Car-Tracking-Data/non-vehicles/*')
    for folder in notcars_images:
        notcars += glob.glob('{}/*.png'.format(folder))

    print("cars size:", len(cars))
    print("notcars size:", len(notcars))
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 8750
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)
    hist_bins = 32
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    # parameters as a dictionary
    parameters = {'color_space': color_space,
                  'orient': orient,
                  'pix_per_cell': pix_per_cell,
                  'cell_per_block': cell_per_block,
                  'hog_channel': hog_channel,
                  'spatial_size': spatial_size,
                  'hist_bins': hist_bins,
                  'spatial_feat': spatial_feat,
                  'hist_feat': hist_feat,
                  'hog_feat': hog_feat}

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Save the data for easy access
    pickle_file = './../../Car-Tracking-Data/features.pickle'
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': X_train,
                        'train_labels': y_train,
                        'test_dataset': X_test,
                        'test_labels': y_test,
                        'X_scaler': X_scaler,
                        'parameters': parameters
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

if save_model:
    # Reload the data
    pickle_file = './../../Car-Tracking-Data/features.pickle'
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
    svc = LinearSVC(max_iter=20000)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    # Save the model for easy access
    pickle_file = './../../Car-Tracking-Data/svc.pickle'
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'svc': svc,
                        'X_scaler': X_scaler,
                        'parameters': parameters
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
