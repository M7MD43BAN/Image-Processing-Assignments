import cv2
import numpy as np


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
# Create a function for adding image to another image
# First parameter: Input image that will be the original
# Second parameter: Input image that will be the added image as a watermark
def subtract_images(first_image, second_image):
    [row, column, channel] = first_image.shape

    # Make sure the two images have the same dimensions
    second_image = cv2.resize(second_image, (first_image.shape[1], first_image.shape[0]))

    # Convert the images to NumPy arrays
    first_image = np.array(first_image, dtype=np.uint8)
    second_image = np.array(second_image, dtype=np.uint8)

    # Create a new NumPy array with the same dimensions as the images
    new_image = np.zeros((first_image.shape[0], first_image.shape[1], 3), dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_image[i, j, k] = np.abs(first_image[i, j, k] - second_image[i, j, k])

    new_image = contrast_adjustment(new_image, new_min=0, new_max=255)

    return new_image

###############################################################################################

original_image = cv2.imread("Squidward.jpeg")
abstracted_image = cv2.imread("logo.png")

new_img = subtract_images(original_image, abstracted_image)

cv2.imshow('Original Image', original_image)
cv2.imshow('Added Image', abstracted_image)
cv2.imshow('New Image', new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
