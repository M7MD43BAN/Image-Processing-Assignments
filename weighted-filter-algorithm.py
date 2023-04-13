import cv2
import numpy as np


def gaussian_kernel(size, sigma):
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 + 1))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernel / np.sum(kernel)


def gaussian_filter(image, size, sigma):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.float32)

    kernel = gaussian_kernel(size, sigma)

    padding = size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    # Apply the filter to each pixel in the image
    for i in range(row):
        for j in range(column):
            neighbor_pixels = padded_image[i:i + size, j:j + size]
            weighted_pixels = neighbor_pixels * kernel

            new_image[i, j] = np.sum(weighted_pixels)

    return new_image.astype(np.uint8)


rgb_image = cv2.imread('Smilling-Shiba-Pics.jpeg', cv2.IMREAD_COLOR)
filtered_rgb_image = cv2.merge([gaussian_filter(rgb_image[:, :, i], 7, 1.5) for i in range(3)])

cv2.imshow('Original Image', rgb_image)
cv2.imshow('Filtered Image', filtered_rgb_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
