import numpy as np
import cv2


###############################################################################################
# Create a function for power law transformations
# First parameter: Input image that will be adjustment
# Second parameter: gamma value
def power_law(image, gamma):
    [row, column, channel] = image.shape
    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_value = int(((image[i, j, k] / 255.0) ** gamma) * 255.0)
                new_image[i, j, k] = new_value
    return new_image


###############################################################################################


img = cv2.imread("Squidward.jpeg")
power_low_img = power_law(img, 1.7)

cv2.imshow('Original Image', img)
cv2.imshow('Adjustment Image', power_low_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
