import numpy as np
import cv2


###############################################################################################
# Create a function for contrast adjustment (histogram stretch/shrink)
# First parameter: Input image that will be adjustment
# Second parameter: New value as maximum
# Third parameter: New value as minimum
def contrast_adjustment(image, new_min, new_max):
    [row, column, channel] = image.shape

    # Find the minimum and maximum intensity values across the image.
    old_min = np.amin(image, axis=(0, 1))
    old_max = np.amax(image, axis=(0, 1))

    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_value = ((image[i, j, k] - old_min[k]) / (old_max[k] - old_min[k])) * (new_max - new_min) + new_min
                if new_value > 255:
                    new_value = 255
                if new_value < 0:
                    new_value = 0
                new_image[i, j, k] = new_value

    return new_image


###############################################################################################

img = cv2.imread("Squidward.jpeg")
contrast_img = contrast_adjustment(img, 50, 200)

cv2.imshow('Original Image', img)
cv2.imshow('Adjustment Image', contrast_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
