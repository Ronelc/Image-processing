import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import skimage
from skimage import color
from scipy import ndimage
import os

BASIC_KERNEL = [1, 1]
CHANGE_FACTOR = 2
MIN_DIM = 16
RGB = "viridis"


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


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    blur_filter = np.multiply(2, blur_filter)
    x_len, y_len = im.shape
    new_rows = np.zeros((CHANGE_FACTOR * x_len, y_len))
    new_rows[::CHANGE_FACTOR] = im
    filtered_rows = ndimage.filters.convolve(new_rows.T, blur_filter,
                                             mode="mirror").T

    x_len, y_len = filtered_rows.T.shape
    new_cols = np.zeros((CHANGE_FACTOR * x_len, y_len))
    new_cols[::CHANGE_FACTOR] = filtered_rows.T
    filtered_im = ndimage.filters.convolve(new_cols.T, blur_filter,
                                           mode="mirror")
    return filtered_im


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    G, org_filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    filter_vec = org_filter_vec.copy()
    pyr = []
    for i in range(0, len(G) - 1):
        pyr.append(G[i] - expand(G[i + 1], filter_vec))
    pyr.append(G[-1])
    return pyr, np.array(filter_vec)


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    for i in range(len(coeff)):
        lpyr[i] = np.multiply(coeff[i], lpyr[i])
    i = len(lpyr) - 1
    while i > 0:
        lpyr[i - 1] += expand(lpyr[i], filter_vec)
        i -= 1
    return lpyr[0]


def stretch_values(im):
    """
    Stretches an image's values to [0,1].
    :param im: an image in float64
    :return: The stretched image
    """
    min_, max_ = im.min(), im.max()
    stretched_im = im - min_
    if max_ != min_:
        stretched_im /= (max_ - min_)
    stretched_im[stretched_im < 0] = 0
    stretched_im[stretched_im > 1] = 1
    return stretched_im


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    pyr = [stretch_values(i) for i in pyr]
    result = pyr[0].copy()
    for i in range(1, levels):
        black_im = np.zeros(
            (result.shape[0] - pyr[i].shape[0], pyr[i].shape[1]))
        new_level = np.concatenate((pyr[i], black_im))
        result = np.concatenate((result, new_level), axis=1)
    return result


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    mask = mask.astype(np.float64)
    L_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    G_m = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    L_out = []
    for k in range(len(L_1)):
        L_out.append(np.multiply(G_m[k], L_1[k]) + np.multiply((1 - G_m[k]),
                                                               L_2[k]))
    return_im = laplacian_to_image(L_out, filter_vec, [1] * len(L_out))
    return_im[return_im < 0] = 0
    return_im[return_im > 1] = 1
    return return_im


def relpath(filename):
    """
     Returns the full path for a given filename with relative path.
    :param filename: a file name.
    :return: The full path
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending(im1, im2, mask, plt_title):
    """
    Perform pyramid blending on two images RGB and a mask
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param plt_title: title for plot
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    max_levels, filter_size = 10, 7
    blended_im = np.dstack(([pyramid_blending(im1[:, :, i], im2[:, :, i], mask,
                                              max_levels, filter_size,
                                              filter_size) for i in range(3)]))

    # create a plot
    fig = plt.figure()
    fig.suptitle(plt_title)
    plt_lst = [(im1, "Image 1", RGB), (im2, "Image 2", RGB),
               (mask, "Mask", "gray"), (blended_im, "Blended image", RGB)]
    for i, plt_args in enumerate(plt_lst):
        p1 = plt.subplot(2, 2, i + 1)
        p1.title.set_text(plt_args[1])
        plt.imshow(plt_args[0], cmap=plt_args[2])
    plt.show()
    return im1, im2, mask, blended_im


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("externals/sword.jpg"), 2)[:400]
    im2 = read_image(relpath("externals/fish.jpg"), 2)[24:424]
    mask = read_image(relpath("externals/mask1.jpg"), 1)
    return blending(im1, im2, mask.astype(bool), "Swordfish")


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("externals/office.jpg"), 2)
    im2 = read_image(relpath("externals/dog.jpg"), 2)
    mask = read_image(relpath("externals/mask2.jpg"), 1)
    return blending(im2, im1, mask.astype(bool), "Dog man")

