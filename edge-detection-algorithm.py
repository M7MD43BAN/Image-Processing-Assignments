import cv2
import numpy as np


def edge_detection(image):
    [row, column] = image.shape

    new_image1 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image2 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image3 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image4 = np.zeros((row, column, 1), dtype=np.uint8)

    # Line Edge Detection
    horizontal_filter = [[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]

    vertical_filter = [[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]]

    diagonal1_filter = [[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]]

    diagonal2_filter = [[2, 1, 0],
                        [1, 0, -1],
                        [0, -1, -2]]

    padding = 1
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Apply each filter to the window and calculate the gradient
            horizontal_gradient = np.sum(np.multiply(mask, horizontal_filter))
            vertical_gradient = np.sum(np.multiply(mask, vertical_filter))
            diagonal1_gradient = np.sum(np.multiply(mask, diagonal1_filter))
            diagonal2_gradient = np.sum(np.multiply(mask, diagonal2_filter))

            # Set the value of the new images based on the gradient values
            new_image1[i - padding, j - padding] = np.abs(horizontal_gradient)
            new_image2[i - padding, j - padding] = np.abs(vertical_gradient)
            new_image3[i - padding, j - padding] = np.abs(diagonal1_gradient)
            new_image4[i - padding, j - padding] = np.abs(diagonal2_gradient)

    # Combine the new images into one image using the maximum gradient
    final_image = np.maximum(np.maximum(np.maximum(new_image1, new_image2), new_image3), new_image4)

    final_image = final_image.astype(np.uint8)

    return final_image


img = cv2.imread('Squidward.jpeg', cv2.IMREAD_GRAYSCALE)
filtered_image = edge_detection(img)

cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
