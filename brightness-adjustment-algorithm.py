import numpy as np
import cv2


###############################################################################################
# Create a function for brightness adjustment
# First parameter: Input image that will be adjustment
# Second parameter: Offset that specify you want brightness or darkness image
def brightness_adjustment(image, offset):
    [row, column, channel] = image.shape
    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_value = image[i, j, k] + offset
                if new_value > 255:
                    new_value = 255
                if new_value < 0:
                    new_value = 0
                new_image[i, j, k] = new_value
    return new_image


###############################################################################################

img = cv2.imread("Squidward.jpeg")
gray_img = brightness_adjustment(img, 50)

cv2.imshow('Original Image', img)
cv2.imshow('Adjustment Image', gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
