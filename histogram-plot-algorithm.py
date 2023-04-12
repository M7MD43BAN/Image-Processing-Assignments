import cv2
import matplotlib.pyplot as plt
import numpy as np


###############################################################################################
# Create a function for histogram plot
# First parameter: Input image that will be adjustment
def histogram_plot(image):
    global histogram

    # Calculate histogram values for grayscale image
    if len(image.shape) == 2:
        histogram = np.zeros(256, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1

        # Draw plot histogram
        plt.bar(range(256), histogram, color='k')
        # The plt.xlim() function sets the x-axis limits to be between 0 and 256
        plt.xlim([0, 256])
        plt.suptitle('Histogram of Gray Image')
        plt.show()

    # Calculate histogram values for RGB color image
    elif len(image.shape) == 3:
        [row, column, channel] = image.shape
        histogram = np.zeros((256, channel), dtype=np.uint8)
        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    histogram[image[i, j, k], k] += 1

        # Draw plot histogram
        # plt.subplots(): specifying 1 row and 3 columns to create a grid of 3 subplots.
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        colors = ['Blue', 'Green', 'Red']
        for i, color in enumerate(colors):
            axs[i].bar(range(256), histogram[:, i], color=color)
            axs[i].set_xlim([0, 256])

        plt.suptitle('Histogram of RGB Image')
        plt.show()


###############################################################################################

img = cv2.imread("Squidward.jpeg")
histogram_plot(img)
