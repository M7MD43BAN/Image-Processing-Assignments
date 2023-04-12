import cv2
import matplotlib.pyplot as plt
import numpy as np


###############################################################################################
def histogram_equalization(image):
    global histogram

    # Calculate histogram values for grayscale image
    if len(image.shape) == 2:
        histogram = np.zeros(8, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1

        # Calculate running sum
        running_sum = np.zeros(8, dtype=int)
        running_sum[0] = histogram[0]
        for i in range(1, 8):
            running_sum[i] = running_sum[i - 1] + histogram[i]

        # Calculate histogram equalization
        pixels_sum = histogram.sum()
        equalized_values = np.zeros(8, dtype=int)
        for i in range(8):
            equalized_values[i] = int(round((7 * running_sum[i]) / pixels_sum))

        # Draw histogram equalization
        plt.bar(range(8), equalized_values, color='k')
        # The plt.xlim() function sets the x-axis limits to be between 0 and 256
        plt.xlim([0, 8])
        plt.suptitle('Histogram of Gray Image')
        plt.show()


###############################################################################################

arr = np.array([[0, 0, 0, 1],
                [1, 1, 1, 2],
                [2, 2, 2, 3],
                [5, 6, 4, 3]])
histogram_equalization(arr)
