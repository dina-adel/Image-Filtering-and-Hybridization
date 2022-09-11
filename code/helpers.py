# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import scipy.fftpack as fp
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy import signal
import cv2


def my_imfilter(image: np.ndarray, filter: np.ndarray, padding_method):
    """ Apply a filter to an input image
    """
    orig_image = image
    filtered_image = np.zeros_like(image)

    # Check the filter size:
    size = filter.shape
    if size[0] % 2 != 0 | size[1] % 2 != 0:
        print('Wrong Filter Size! The size should be odd')
        return

    # Flip the filter
    filter = np.flipud(np.fliplr(filter))
    size = filter.shape
    extra_rows = size[0] // 2
    extra_cols = size[1] // 2

    if padding_method == 'zero':
        # Pad the image with Zeros
        padded_image = np.pad(image, ((extra_rows, extra_rows), (extra_cols, extra_cols), (0, 0)), 'constant',
                              constant_values=(0))

    elif padding_method == 'reflect':
        padded_image = np.pad(image, ((extra_rows, extra_rows), (extra_cols, extra_cols), (0, 0)), 'reflect')

    # Applying the filter
    if orig_image.shape[2] == 3:
        filter3d = np.repeat(filter[:, :, np.newaxis], 3, axis=2)

    for color in range(orig_image.shape[2]):
        for x in range(orig_image.shape[0]):
            for y in range(orig_image.shape[1]):
                filtered_image[x, y, color] = np.sum(
                    (filter3d[:, :, color] * padded_image[x:x + size[0], y:y + size[1], color]))

    x = 2
    return filtered_image


def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float, k_size, padding_method):
    """
   Inputs:
       - image1 -> The image from which to take the low frequencies.
       - image2 -> The image from which to take the high frequencies.
       - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                             blur that will remove high frequencies.
       -padding_method-> zero pad or reflect pad
   """
    assert image1.shape == image2.shape

    kernel = np.dot(cv2.getGaussianKernel(k_size, cutoff_frequency), cv2.getGaussianKernel(k_size, cutoff_frequency).T)

    low_frequencies = my_imfilter(image1, kernel, padding_method)  # dog
    high_frequencies = image2 - my_imfilter(image2, kernel, padding_method)  # cat

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = low_frequencies + high_frequencies

    hybrid_image = np.clip(hybrid_image, 0, 1)
    return low_frequencies, high_frequencies, hybrid_image


def vis_hybrid_image(hybrid_image: np.ndarray):
    """ Visualize a hybrid image by progressively down-sampling the image and
        concatenating all the images together.
    """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # down-sample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output


def load_image(path):
    return img_as_float32(io.imread(path))


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))
