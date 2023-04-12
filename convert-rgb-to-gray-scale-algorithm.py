import cv2
import numpy as np


###############################################################################################
# Create a function for converting RGB image to Gray image
# First parameter: Input image that will be converted

def rgb2gray(rgb_image):
    # Convert the RGB image to grayscale using the formula: gray = 0.3RED + 0.59GREEN + 0.11BLUE
    gray_image = (0.3 * rgb_image[:, :, 0]) + (0.3 * rgb_image[:, :, 1]) + (0.3 * rgb_image[:, :, 2])

    return gray_image.astype(np.uint8)


###############################################################################################

# imread(): this method loads an image from the specified file
img = cv2.imread("Squidward.jpeg")
gray_img = rgb2gray(img)

# Displaying the original and resized images using the imshow function of OpenCV
cv2.imshow('RGB Image', img)
print(img.shape)

cv2.imshow('Gray Image', gray_img)
print(gray_img.shape)

# hold the screen until user close it.
cv2.waitKey(0)

# Deleting created GUI window from screen and memory
cv2.destroyAllWindows()
