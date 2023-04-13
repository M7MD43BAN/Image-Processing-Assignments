import cv2
import numpy as np


def average_filter(image, mask_size):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.uint8)

    padding = mask_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            mask_mean = np.sum(mask) / (mask_size ** 2)
            new_image[i - padding, j - padding] = mask_mean

    return new_image


gray_image = cv2.imread('Squidward.jpeg', cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.imread('Squidward.jpeg', cv2.IMREAD_COLOR)

filtered_gray_image = average_filter(gray_image, 7)
filtered_rgb_image = cv2.merge([average_filter(rgb_image[:, :, i], 7) for i in range(3)])

cv2.imshow('Original Gray Image', gray_image)
cv2.imshow('Original RGB Image', rgb_image)
cv2.imshow('Filtered Image', filtered_rgb_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
