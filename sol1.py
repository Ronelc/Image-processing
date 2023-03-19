import numpy as np
import imageio as iio
import skimage
from skimage import color
# import scipy
from matplotlib import pyplot as plt

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
MAX_VAL = 255


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as a np.float64 matrix normalized to [0,1]
    """
    img = iio.imread(filename)

    org_img = GRAYSCALE if len(img.shape) == 2 else RGB

    if representation == GRAYSCALE:
        if org_img == GRAYSCALE:
            return np.float64(img / MAX_VAL)
        else:
            return skimage.color.rgb2gray(img)
    else:
        return np.float64(img / MAX_VAL)


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    color_map = 'gray' if representation == GRAYSCALE else 'viridis'
    plt.imshow(read_image(filename, representation), interpolation='nearest',
               cmap=color_map)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    return (imRGB @ RGB_YIQ_TRANSFORMATION_MATRIX.T).reshape(imRGB.shape)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    return (imYIQ @ np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX).T).reshape(
        imYIQ.shape)


def return_to_RGB(im_color, channel_eq, YIQ):
    if im_color == RGB:
        new_im_yiq = np.dstack(
            (channel_eq / MAX_VAL, YIQ[:, :, 1], YIQ[:, :, 2]))
        return yiq2rgb(new_im_yiq)
    else:
        return channel_eq / MAX_VAL


def find_Y_channel(im_color, im_orig):
    YIQ = None
    to_equalized = im_orig
    if im_color == RGB:
        YIQ = rgb2yiq(im_orig)
        to_equalized = YIQ[:, :, 0]
    to_equalized = (to_equalized * MAX_VAL).astype(int)
    return to_equalized, YIQ


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    # is the original picture grayscale or RGB
    im_color = GRAYSCALE if len(im_orig.shape) == 2 else RGB

    # if RGB - find Y
    to_equalized, YIQ = find_Y_channel(im_color, im_orig)

    hist_orig, x = np.histogram(to_equalized, bins=np.arange(MAX_VAL + 2))

    m = hist_orig.nonzero()[0][0]

    C = np.cumsum(hist_orig)

    hist_eq = np.round((C / C[-1]) * MAX_VAL)

    if C[-1] != C[m]:
        hist_eq = np.round(((C - C[m]) / (C[-1] - C[m])) * MAX_VAL)

    channel_eq = hist_eq[to_equalized].astype(int)

    return [return_to_RGB(im_color, channel_eq, YIQ), hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    # is the original picture grayscale or RGB
    im_color = GRAYSCALE if len(im_orig.shape) == 2 else RGB
    to_quantize, YIQ = find_Y_channel(im_color, im_orig)  # if RGB - find Y
    hist_orig, X1 = np.histogram(to_quantize, bins=np.arange(MAX_VAL + 2))
    normalized_hist = hist_orig / np.sum(hist_orig)

    q, error, Z = [0] * n_quant, [0] * n_iter, [0]
    cum_hist = np.cumsum(hist_orig)
    factor = cum_hist[MAX_VAL] // n_quant
    for i in range(1, n_quant):
        Z.append(np.where(cum_hist > i * factor)[0][0])
    Z.append(MAX_VAL)

    last_ndx = n_iter
    for iter_ in range(n_iter):

        # find optimal q
        for i in range(n_quant):
            z_0, z_1 = np.floor(Z[i]).astype(int), np.floor(Z[i + 1]).astype(
                int)
            h_g = normalized_hist[z_0:z_1]
            q[i] = 0 if np.sum(h_g) == 0 else np.sum(
                np.arange(z_0, z_1) * h_g) / np.sum(h_g)

        # find the optimal Z
        for i in range(1, n_quant):
            Z[i] = (q[i - 1] + q[i]) / 2

        # calculate the error
        array = np.arange(MAX_VAL + 1).astype(float)
        for i in range(n_quant):
            z_0, z_1 = np.floor(Z[i]).astype(int), np.floor(Z[i + 1]).astype(
                int)
            if len(array) <= 1: break
            array[z_0:z_1] -= q[i]
        array[MAX_VAL] -= q[n_quant - 1]
        error[iter_] = np.sum(np.multiply(np.square(array), normalized_hist))

        if iter_ > 0 and error[iter_] == error[iter_ - 1]:
            last_ndx = iter_
            break
    error = error[0:last_ndx]

    # create the new image
    quantize_channel = np.copy(hist_orig)
    for i in range(n_quant):
        z_0, z_1 = np.floor(Z[i]).astype(int), np.floor(Z[i + 1]).astype(int)
        quantize_channel[z_0:z_1] = q[i]
    quantize_channel[MAX_VAL] = q[n_quant - 1]
    quantize_channel = quantize_channel[to_quantize.astype(int)]
    return [return_to_RGB(im_color, quantize_channel, YIQ), error]


im_quant = []


def split_into_buckets(im_orig, img_array, n_quant):
    if n_quant == 0:
        rgb_average_lst = [0, 0, 0]
        for i, rgb in enumerate(rgb_average_lst):
            rgb_average_lst[i] = np.mean(img_array[:, i])
        for data in img_array:
            data = data.astype(int)
            im_quant[data[3]][data[4]] = rgb_average_lst
        return

    rgb_range_lst = [0, 0, 0]
    for i, rgb in enumerate(rgb_range_lst):
        if len(img_array) == 0: break
        rgb_range_lst[i] = np.max(img_array[:, i]) - np.min(img_array[:, i])
    space_with_highest_range = rgb_range_lst.index(max(rgb_range_lst))

    img_array = img_array[img_array[:, space_with_highest_range].argsort()]
    median_index = len(img_array) // 2

    split_into_buckets(im_orig, img_array[:median_index], n_quant - 1)
    split_into_buckets(im_orig, img_array[median_index:], n_quant - 1)


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]f
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    global im_quant
    im_quant = np.copy(im_orig)
    img_array = []
    for rindex, rows in enumerate(im_orig):
        for cindex, rgb in enumerate(rows):
            img_array.append(
                [rgb[0], rgb[1], rgb[2], rindex, cindex])

    split_into_buckets(im_quant, np.array(img_array), n_quant)
    return im_quant / MAX_VAL

