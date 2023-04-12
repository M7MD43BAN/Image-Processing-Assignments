import cv2
import numpy as np


###############################################################################################
# Create a function for adding image to another image
# First parameter: Input image that will be the original
# Second parameter: Input image that will be the added image as a watermark
def negative_image(image):
    [row, column, channel] = image.shape
    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_image[i, j, k] = 255 - image[i, j, k]

    return new_image


###############################################################################################

original_image = cv2.imread("Squidward.jpeg")

new_img = negative_image(original_image)

cv2.imshow('Original Image', original_image)
cv2.imshow('Negative Image', new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
