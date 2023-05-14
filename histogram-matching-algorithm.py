import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_plot(image):
    global histogram

    # Calculate histogram values for grayscale image
    if len(image.shape) == 2:
        histogram = np.zeros(256, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1

    # Calculate histogram values for RGB color image
    elif len(image.shape) == 3:
        [row, column, channel] = image.shape
        histogram = np.zeros((256, channel), dtype=np.uint8)
        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    histogram[image[i, j, k], k] += 1

    return histogram


def histogram_matching(first_image, second_image):
    # Compute histograms of the input images
    histogram1 = histogram_plot(first_image)
    histogram2 = histogram_plot(second_image)

    # Compute cumulative distribution functions (CDFs) of the input images
    cumulative_distribution_function1 = np.cumsum(histogram1) / first_image.size
    cumulative_distribution_function2 = np.cumsum(histogram2) / second_image.size

    # Create a lookup table to map intensity levels from img to img2
    map_array = np.zeros((256,), dtype=np.uint8)
    for i in range(256):
        diff = np.abs(cumulative_distribution_function1[i] - cumulative_distribution_function2)
        ind = np.argmin(diff)
        map_array[i] = ind

    # Apply the mapping function to the input image
    new_image = map_array[first_image]

    # Compute histogram of the output image
    new_image_histogram = histogram_plot(new_image)

    return new_image, new_image_histogram


img = cv2.imread("pic1.jpg", 0)
img2 = cv2.imread("pic2.jpeg", 0)

new_output_image, new_output_image_histogram = histogram_matching(img, img2)

# Display input images, output image, and histograms
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].imshow(img, cmap="gray")
axs[0, 0].set_title("img")

axs[0, 1].imshow(img2, cmap="gray")
axs[0, 1].set_title("img2")

axs[0, 2].imshow(new_output_image, cmap="gray")
axs[0, 2].set_title("Matched")

axs[1, 0].bar(range(256), histogram_plot(img), color='k')
axs[1, 0].set_title("Histogram of image 1")

axs[1, 1].bar(range(256), histogram_plot(img2), color='k')
axs[1, 1].set_title("Histogram of image 2")

axs[1, 2].bar(range(256), new_output_image_histogram, color='k')
axs[1, 2].set_title("Histogram of new image")

plt.show()
