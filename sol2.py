import numpy as np
from scipy import signal
import scipy.io.wavfile
from scipy.ndimage.interpolation import map_coordinates
import skimage
from skimage import color
import imageio as iio

GRAYSCALE = 1
RGB = 2
MAX_VAL = 255
FOURIER_FACTOR = 2 * np.pi * 1j


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


def dft_basic_formula(signal, formula_args):
    """

    :param signal: array of dtype float64 with shape (N,) or (N,1)
    :param formula_args: formula argument multiply by -1 and divide by N if DFT
            multiply by 1 and divide by 1 id IDFT.
    :return:
    """
    N = signal.shape[0]
    f_x = np.identity(N)
    x = np.arange(N)
    u = np.meshgrid(x, x)[1]
    pow = (formula_args[0] * FOURIER_FACTOR * u * x) / N
    F_u = (f_x @ np.exp(pow)) / formula_args[1]
    return F_u.dot(signal)


def DFT(signal):
    """
    Transform a 1D discrete signal to its Fourier representation
    :param signal: array of dtype float64 with shape (N,) or (N,1)
    :return: fourier_signal - complex signal, array of dtype complex128 with the same shape
    """
    fourier_signal = dft_basic_formula(signal, [-1, 1])
    return fourier_signal.astype(np.complex128)


def IDFT(fourier_signal):
    """
    Transform a 1D Fourier signal to its discrete representation
    :param fourier_signal: array of dtype complex128 with shape (N,) or (N,1)
    :return: signal: complex Fourier signal
    """
    signal = dft_basic_formula(fourier_signal, [1, fourier_signal.shape[0]])
    return signal.astype(np.complex128)


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of type float64, shape (M,N) or (M,N,1).
    :return: Fourier representation
    """
    shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape(shape[0], shape[1])
    rows_dft = DFT(image)
    return DFT(rows_dft.T).T.reshape(shape)


def IDFT2(fourier_image):
    """
    convert a 2D Fourier representation signal to its discrete representation
    :param fourier_image: 2D array of type complex128, shape (M,N) or (M,N,1).
    :return: discrete representation
    """
    shape = fourier_image.shape
    if len(fourier_image.shape) == 3:
        fourier_image = fourier_image.reshape(shape[0], shape[1])
    rows_idft = IDFT(fourier_image)
    return IDFT(rows_idft.T).T.reshape(shape)


def change_rate(filename, ratio):
    """
    function that changes the duration of an audio file by keeping the
    same samples, but changing the sample rate written in the file header.
    :param filename: string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change.
           0.25 < ratio < 4
    :return: no return
    """
    sample_rate, sample = scipy.io.wavfile.read(filename)
    new_sample_rate = int(sample_rate * ratio)
    scipy.io.wavfile.write("change_rate.wav", new_sample_rate, sample)


def resize(data, ratio):
    """
    Change the number of samples by the given ratio.
    :param data: 1D ndarray of type float64 or complex128(*)
           representing the original sample points
    :param ratio: ratio: positive float64 representing the duration change.
           0.25 < ratio < 4
    :return: 1D ndarray of the type of data representing the new sample points.
    """
    # find the new length
    num_of_samples = len(data)
    if ratio > 1:
        if ratio % 2 == 0:
            new_len = np.ceil(num_of_samples * (1 - 1 / ratio))
        else:
            new_len = num_of_samples - np.ceil(num_of_samples / ratio)
    else:
        new_len = np.floor(num_of_samples * ((1 / ratio) - 1))
    left = int(new_len / 2)
    right = left if new_len % 2 == 0 else left + 1

    if ratio < 1:  # case of slowing down.
        # add zero to high frequency
        new_dft = np.pad(DFT(data), (left, right), 'constant')
    else:  # case of Fast forward
        new_dft = np.fft.fftshift(DFT(data))[left:num_of_samples - right]
        new_dft = np.fft.ifftshift(new_dft)
    return IDFT(new_dft).astype(data.dtype)


def change_samples(filename, ratio):
    """
    function that changes the duration of an audio file by reducing the
    number of samples using Fourier
    :param filename: string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change.
           0.25 < ratio < 4
    :return: 1D ndarray of type float64 representing the new sample points
    """
    sample_rate, sample = scipy.io.wavfile.read(filename)
    new_sample = resize(sample, ratio)
    scipy.io.wavfile.write("change_samples.wav", sample_rate, new_sample)
    return new_sample.astype(np.float64)


def resize_spectrogram(data, ratio):
    """
    function that speeds up a WAV file, without changing the pitch,
    using spectrogram scaling. This is done by computing the spectrogram,
    changing the number of spectrogram columns, and creating back the audio.
    :param data: 1D ndarray of dtype float64 representing the original
           sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data
    """
    spectrogram = stft(data)
    new_spectrogram = [resize(row, ratio) for row in spectrogram]
    return istft(np.array(new_spectrogram)).astype(data.dtype)


def resize_vocoder(data, ratio):
    """
    function that speedups a WAV file by phase vocoding its spectrogram.
    :param data: 1D ndarray of dtype float64 representing the original sample points.
    :param ratio: positive float64 representing the rate change of the WAV file.
    :return: the given data rescaled according to ratio with the same datatype as data.
    """
    spectrogram = stft(data)
    warped_spec = phase_vocoder(spectrogram, ratio)
    return istft(warped_spec).astype(data.dtype)


def conv_der(im):
    """
    function that computes the magnitude of image derivatives.
    :param im: grayscale images of type float64
    :return: grayscale images of type float64, magnitude
            of the derivative, with the same dtype and shape of im.
    """
    kernel = np.array([[0.5, 0, -0.5]])
    dx = scipy.signal.convolve2d(im, kernel, mode='same')
    dy = scipy.signal.convolve2d(im, kernel.T, mode='same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude.astype(im.dtype).reshape(im.shape)


def fourier_der(im):
    """
    computes the derivatives of an image in both axis, using fourier
    :param im: greyscale image of type float64
    :return: greyscale image of type float64 - magnitude of derivatives
    """
    N, M = im.shape
    u = FOURIER_FACTOR * np.arange(-N / 2, N / 2) / N
    v = FOURIER_FACTOR * np.arange(-M / 2, M / 2) / M
    im_f = DFT2(im)
    shifted_dft = np.fft.fftshift(im_f)
    dx = IDFT2(np.fft.ifftshift((u * shifted_dft.T).T))
    dy = IDFT2(np.fft.ifftshift(v * shifted_dft))
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude.astype(np.float64)


######################### school's code ############################

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec



