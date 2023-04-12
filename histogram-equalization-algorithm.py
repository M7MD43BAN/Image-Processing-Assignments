import cv2
import matplotlib.pyplot as plt
import numpy as np


###############################################################################################
# Create a function for histogram plot
# First parameter: Input image that will be adjustment
def histogram_equalization(image):
    global histogram

    # Calculate histogram values for grayscale image
    if len(image.shape) == 2:
        histogram = np.zeros(256, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1

        # Calculate running sum
        running_sum = np.zeros(256, dtype=int)
        running_sum[0] = histogram[0]
        for i in range(1, 256):
            running_sum[i] = running_sum[i - 1] + histogram[i]

        # Calculate histogram equalization
        pixels_sum = histogram.sum()
        equalized_values = np.zeros(256, dtype=int)
        for i in range(256):
            equalized_values[i] = int(round((255 * running_sum[i]) / pixels_sum))

        # Draw histogram equalization
        plt.bar(range(256), equalized_values, color='k')
        # The plt.xlim() function sets the x-axis limits to be between 0 and 256
        plt.xlim([0, 256])
        plt.suptitle('Histogram of Gray Image')
        plt.show()


###############################################################################################

img = cv2.imread("Squidward.jpeg", 0)
histogram_equalization(img)
