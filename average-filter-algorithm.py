import cv2
import numpy as np


def average_filter(image, mask_size):
    [row, column, channel] = image.shape
    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    padding = mask_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Extract the mask from the padded image
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Calculate the mean value of the mask - axis=(0, 1) for the channels
            mask_mean = np.sum(mask, axis=(0, 1)) / (mask_size ** 2)

            # Set the corresponding pixel in the filtered image to the mean value
            new_image[i - padding, j - padding] = mask_mean

    return new_image


img = cv2.imread("Squidward.jpeg")
filtered_img = average_filter(img, 7)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
