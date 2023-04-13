import cv2
import numpy as np


def sharpening(image):
    [row, column] = image.shape

    new_image1 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image2 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image3 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image4 = np.zeros((row, column, 1), dtype=np.uint8)

    # Line Edge Detection
    horizontal_filter = [[0, 1, 0],
                         [0, 1, 0],
                         [0, -1, 0]]

    vertical_filter = [[0, 0, 0],
                       [1, 1, -1],
                       [0, 0, 0]]

    diagonal1_filter = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]]

    diagonal2_filter = [[0, 0, 1],
                        [0, 1, 0],
                        [-1, 0, 0]]

    padding = 1
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Convolution
            window = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Horizontal Filter
            horizontal_result = (window * horizontal_filter).sum()
            new_image1[i - padding, j - padding] = abs(horizontal_result)

            # Vertical Filter
            vertical_result = (window * vertical_filter).sum()
            new_image2[i - padding, j - padding] = abs(vertical_result)

            # Diagonal Filter 1
            diagonal1_result = (window * diagonal1_filter).sum()
            new_image3[i - padding, j - padding] = abs(diagonal1_result)

            # Diagonal Filter 2
            diagonal2_result = (window * diagonal2_filter).sum()
            new_image4[i - padding, j - padding] = abs(diagonal2_result)

    # combining four different line edge detection filters (horizontal, vertical, and two diagonal filters)
    # using bitwise OR operations.
    sharpened_image = cv2.bitwise_or(new_image1, new_image2)
    sharpened_image = cv2.bitwise_or(sharpened_image, new_image3)
    sharpened_image = cv2.bitwise_or(sharpened_image, new_image4)

    return sharpened_image


img = cv2.imread('Squidward.jpeg', cv2.IMREAD_GRAYSCALE)
filtered_image = sharpening(img)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
