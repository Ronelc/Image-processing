from scipy import ndimage
from scipy.signal import convolve2d
import numpy as np
import imageio as iio
import skimage.color


BASIC_KERNEL = [1, 1]
CHANGE_FACTOR = 2
MIN_DIM = 16


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



################## previous exercises code #######################
def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as a np.float64 matrix normalized to [0,1]
    """
    img = iio.imread(filename)

    org_img = 1 if len(img.shape) == 2 else 2

    if representation == 1:
        if org_img == 1:
            return np.float64(img / 255)
        else:
            return skimage.color.rgb2gray(img)
    else:
        return np.float64(img / 255)


def create_gaussian_kernel(filter_size):
    """
    create Gaussian kernel in size filter_size
    :param filter_size: kernel size
    :return: Gaussian kernel
    """
    kernel = BASIC_KERNEL
    while len(kernel) < filter_size:
        kernel = np.convolve(kernel, BASIC_KERNEL)
    return [list(kernel / sum(kernel))]


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    filtered_rows = ndimage.filters.convolve \
                        (im, blur_filter, mode="mirror")[::CHANGE_FACTOR]
    filtered_im = ndimage.filters.convolve \
                      (filtered_rows.T, blur_filter, mode="mirror")[
                  ::CHANGE_FACTOR]
    return filtered_im.T


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = [im]
    filter_vec = create_gaussian_kernel(filter_size)
    for level in range(1, max_levels):
        x_len, y_len = im.shape
        if x_len / CHANGE_FACTOR > MIN_DIM \
                and y_len / CHANGE_FACTOR > MIN_DIM:
            im = reduce(im, filter_vec)
            pyr.append(im)
    return pyr, np.array(filter_vec)
