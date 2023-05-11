import cv2
import numpy as np


def low_pass_ideal(image, radius):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                if distance > radius:
                    mask[i, j, k] = 0
                else:
                    mask[i, j, k] = 255

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered, mask


original_image = cv2.imread("Squidward.jpeg")
new_img, mask = low_pass_ideal(original_image, 50)

cv2.imshow('Original Image', original_image)
cv2.imshow('Mask', mask)
cv2.imshow('Filtered Image', new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
