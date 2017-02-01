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
image = mpimg.imread('./../../Car-Tracking-Data/examples/test5.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255


# Read in cars and notcars
cars = []
notcars = []

cars_images = glob.glob('./../../Car-Tracking-Data/vehicles/*')
for folder in cars_images:
    cars += glob.glob('{}/*.png'.format(folder))

notcars_images = glob.glob('./../../Car-Tracking-Data/non-vehicles/*')
for folder in notcars_images:
    notcars += glob.glob('{}/*.png'.format(folder))

# take one from each group
car = cars[500]
notcar = notcars[500]

# load image
image_car = mpimg.imread(car)
image_notcar = mpimg.imread(notcar)

# Construct histograms
def plot_color_hist(original, hist_bins=32):
    image = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
    rhist = np.histogram(image[:,:,0], bins=hist_bins, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=hist_bins, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=hist_bins, range=(0, 256))


    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Plot a figure with all three bar charts
    fig = plt.figure(figsize=(12,3))
    plt.subplot(141)
    plt.imshow(original)
    plt.title('Image')
    plt.subplot(142)
    plt.bar(bin_centers, rhist[0])
    plt.xlim(0, 256)
    plt.title('Y Histogram')
    plt.subplot(143)
    plt.bar(bin_centers, ghist[0])
    plt.xlim(0, 256)
    plt.title('Cr Histogram')
    plt.subplot(144)
    plt.bar(bin_centers, bhist[0])
    plt.xlim(0, 256)
    plt.title('Cb Histogram')
    plt.show()

plot_color_hist(image_car)
plot_color_hist(image_notcar)