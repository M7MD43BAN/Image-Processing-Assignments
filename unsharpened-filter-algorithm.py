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


def subtract_images(first_image, second_image):
    [row, column] = first_image.shape

    # Make sure the two images have the same dimensions
    second_image = cv2.resize(second_image, (first_image.shape[1], first_image.shape[0]))

    # Convert the images to NumPy arrays
    first_image = np.array(first_image, dtype=np.uint8)
    second_image = np.array(second_image, dtype=np.uint8)

    # Create a new NumPy array with the same dimensions as the images
    new_image = np.zeros((first_image.shape[0], first_image.shape[1], 3), dtype=np.uint8)

    for i in range(row):
        for j in range(column):
            new_image[i, j] = np.abs(first_image[i, j] - second_image[i, j])

    new_image = contrast_adjustment(new_image, new_min=0, new_max=255)

    return new_image


def merge_images(first_image, second_image):
    [row, column] = first_image.shape

    # Make sure the two images have the same dimensions
    second_image = cv2.resize(second_image, (first_image.shape[1], first_image.shape[0]))

    # Convert the images to NumPy arrays
    first_image = np.array(first_image, dtype=np.uint8)
    second_image = np.array(second_image, dtype=np.uint8)

    # Create a new NumPy array with the same dimensions as the images
    new_image = np.zeros((first_image.shape[0], first_image.shape[1], 3), dtype=np.uint8)

    for i in range(row):
        for j in range(column):
            new_image[i, j] = first_image[i, j] + second_image[i, j]

    new_image = contrast_adjustment(new_image, new_min=0, new_max=255)

    return new_image


def unsharpened(image):
    gaussian_image = gaussian_filter(image, 5, 2)
    subtract_image = subtract_images(first_image=image, second_image=gaussian_image)

    return merge_images(first_image=image, second_image=subtract_image)


img = cv2.imread('Squidward.jpeg', cv2.IMREAD_GRAYSCALE)
filtered_image = unsharpened(img)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
