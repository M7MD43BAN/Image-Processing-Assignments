import cv2
import numpy as np


###############################################################################################
# Create a function for Resizing Direct Mapping: 1-Order algorithm
# First parameter: Input image that will be resized
# Second parameter: resizing factor along the horizontal and the vertical axes
def direct_mapping_1Order(old_image, factor):
    # Saving the shape width (row) and height (column) and channels
    # of the image using shape attribute.
    [row, column, channel] = old_image.shape

    # calculate the new dimensions of the resized image using `factor`.
    new_row = row * factor
    new_column = column * factor

    # Creating a new matrix of zeros with the new dimensions of the resized image.
    # We use the zeros function of NumPy to create this new matrix.
    # We also specify the data type of the matrix as unsigned 8-bit integers using the dtype argument.
    resized_image = np.zeros([new_row, new_column, channel], dtype=np.uint8)

    # Copying an old values of the old image into the new image that will be resized
    # We iterate over each channel of the image using k
    # Then for each pixel in the image using i and j multiplying by the factor
    for k in range(channel):
        for i in range(row):
            for j in range(column):
                resized_image[i * factor, j * factor, k] = old_image[i, j, k]

    # We iterate throw each row to fill the gaps between the pixels in the resized image.
    # Filling in the rows by comparing adjacent pixel values in each row using linear interpolation
    for k in range(channel):
        for i in range(0, new_row, factor):
            for j in range(0, new_column - factor - 1, factor):
                # Saving the maximum and minimum pixels from the resized image
                minimum = resized_image[i, j, k]
                maximum = resized_image[i, j + factor, k]
                # Case of from top (minimum) to bottom (maximum)
                if maximum > minimum:
                    for pixel in range(1, factor):
                        # Pixel(i) = Round(((Max - Min)/Fact)*i + Min))
                        resized_image[i, j + pixel, k] = round(((maximum - minimum) / factor) * pixel + minimum)
                # Case of from bottom (minimum) to top (maximum)
                else:
                    for pixel in range(1, factor):
                        # Pixel(i) = Round(((Min - Max)/Fact)*i + Max))
                        resized_image[i, j + factor - pixel, k] = round(
                            ((minimum - maximum) / factor) * pixel + maximum
                        )
            # The rest of pixels that not between min and max pixels
            resized_image[i, new_column - factor + 1:new_column, k] = resized_image[i, new_column - factor, k]

    # We iterate throw each column to fill the gaps between the pixels in the resized image.
    # Filling in the column by comparing adjacent pixel values in each column using linear interpolation
    for k in range(channel):
        for j in range(0, new_column):
            for i in range(0, new_row - factor - 1, factor):
                # Saving the maximum and minimum pixels from the resized image
                minimum = resized_image[i, j, k]
                maximum = resized_image[i + factor, j, k]
                # Case of from right (minimum) to left (maximum)
                if maximum > minimum:
                    for pixel in range(1, factor):
                        # Pixel(i)= Round(((Max - Min)/Fact)*i + Min))
                        resized_image[i + pixel, j, k] = int(round(((maximum - minimum) / factor) * pixel + minimum))
                # Case of from left (minimum) to right (maximum)
                else:
                    for pixel in range(1, factor):
                        # Pixel(i)= Round(((Min - Max)/Fact)*i + Max))
                        resized_image[i + factor - pixel, j, k] = round(
                            ((minimum - maximum) / factor) * pixel + maximum
                        )
            # The rest of pixels that not between min and max pixels
            resized_image[new_row - factor:new_row, j, k] = resized_image[new_row - factor, j, k]

    return resized_image


###############################################################################################

# imread(): this method loads an image from the specified file
image = cv2.imread("/home/m7md43ban/Image Processing Assignments/pyramid.jpg")
new_image = direct_mapping_1Order(image, 2)

# Displaying the original and resized images using the imshow function of OpenCV
cv2.imshow('Original Image', image)
print(image.shape)
cv2.imshow('Resized', new_image)
print(new_image.shape)

# hold the screen until user close it.
cv2.waitKey(0)

# Deleting created GUI window from screen and memory
cv2.destroyAllWindows()
