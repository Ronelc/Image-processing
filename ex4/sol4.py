import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, filters
import shutil
from imageio import imwrite
from scipy.ndimage import convolve, map_coordinates

import sol4_utils

WINDOW = lambda r: 2 * r + 1
PYRAMID_MAX_L = 3
DERIVE_KERNEL = [[1, 0, -1]]
KERNEL_SIZE = 3
K_h = 0.04
RADIUS = 3
M, N = 7, 7


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    x_der = convolve(im, DERIVE_KERNEL)
    y_der = convolve(im.T, DERIVE_KERNEL).T

    xx = sol4_utils.blur_spatial(x_der * x_der, KERNEL_SIZE)
    xy = sol4_utils.blur_spatial(x_der * y_der, KERNEL_SIZE)
    yy = sol4_utils.blur_spatial(y_der * y_der, KERNEL_SIZE)

    det_m = (xx * yy) - np.square(xy)
    R = det_m - K_h * np.square(xx + yy)
    return np.flip(np.argwhere(non_maximum_suppression(R)), axis=1)


def get_normalized_matrix(mat):
    """
    normalized a given matrix - represent a window. Useful to be invariant to lighting changes
    :param mat: matrix
    :return: normalized matrix
    """
    var = mat - np.mean(mat)
    if np.linalg.norm(var) != 0:
        return var / np.linalg.norm(var)
    return var


def indexes(x_range, y_range, homography=None):
    """
    create indexes array in window range for a given range arrays
    :param x_range: range of x-axis
    :param y_range: range of y-axis
    :param homography: homography
    :return: indexes array in radius range, shape (2, K * K)
    """
    gx, gy = np.meshgrid(x_range, y_range)
    coordinates = np.dstack((gx, gy))
    n, m, l = coordinates.shape
    coordinates = coordinates.reshape(n * m, l)
    if homography is not None:
        coordinates = apply_homography(coordinates, np.linalg.inv(homography))
    return np.flip(coordinates.T, axis=0), gy.shape


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = WINDOW(desc_rad)
    descriptors_arr = np.zeros((pos.shape[0], K, K))
    for i, (x, y) in enumerate(pos):
        x_range = np.arange(x - desc_rad, x + desc_rad + 1)
        y_range = np.arange(y - desc_rad, y + desc_rad + 1)
        descriptor = map_coordinates(im, indexes(x_range, y_range)[0],
                                     order=1, prefilter=False).reshape(K, K)
        descriptors_arr[i, :, :] = get_normalized_matrix(descriptor)
    return descriptors_arr


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    corners_coordinates = spread_out_corners(pyr[0], M, N, 13)
    descriptor = sample_descriptor(pyr[2], corners_coordinates / 4, RADIUS)
    return corners_coordinates, descriptor


def flatting(desc):
    """
    flatt a feature descriptor array
    :param desc: A feature descriptor array with shape (N,K,K).
    :return: A flatten feature descriptor array with shape (N,K*K).
    """
    return np.array(
        [np.ndarray.flatten(desc[i, :, :]) for i in range(desc.shape[0])])


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    sum_mat = flatting(desc1) @ flatting(desc2).T
    sum_mat[sum_mat <= min_score] = 0

    # create 2 dictionaries with 2 indexes of max values per col / row
    d1, d2 = {}, {}
    for i in range(len(sum_mat[:, 0])):
        d1[i] = list(sum_mat[i].argsort()[-2:])
    for j in range(len(sum_mat[0])):
        d2[j] = list(sum_mat[:, j].argsort()[-2:])

    # find match points
    m1, m2 = [], []
    for k in d1.keys():
        val1, val2 = d1[k]
        if k in d2[val1]:
            m1.append(k)
            m2.append(val1)
        if k in d2[val2]:
            m1.append(k)
            m2.append(val2)
    return np.array(m1), np.array(m2)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    n, m = pos1.shape
    h_coordinates_1 = np.ones((n, m + 1))
    h_coordinates_1[:, :m] = pos1
    new_h_coordinates = (H12 @ h_coordinates_1.T).T
    new_coordinates = np.empty((n, m))
    new_coordinates = new_h_coordinates[:, :m] / new_h_coordinates[:, m:]
    return new_coordinates


def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    inliers = []
    for iter in range(num_iter):
        x1, x2 = np.random.choice(points1.shape[0], size=2)
        p1 = np.array([points1[x1], points1[x2]])
        p2 = np.array([points2[x1], points2[x2]])
        curr_H = estimate_rigid_transform(p1, p2, translation_only)
        new_coordinates = apply_homography(points1, curr_H)
        dist = np.square(np.linalg.norm(new_coordinates - points2, axis=1))
        curr_inliers = np.argwhere(dist < inlier_tol)
        if len(curr_inliers) > len(inliers):
            inliers = np.ndarray.flatten(curr_inliers)
    H = estimate_rigid_transform(points1[inliers], points2[inliers],
                                 translation_only)
    return H, inliers


def plot_marks(arr1, arr2, len, color):
    """
    plot the marks and lines between them
    :param arr1: firs array
    :param arr2: second array
    :param len: weigh of image
    :param color: line color
    """
    for i in range(arr1.shape[0]):
        plt.plot([arr1[:, 0][i:i + 2], (arr2[:, 0] + len)[i:i + 2]],
                 [arr1[:, 1][i:i + 2], arr2[:, 1][i:i + 2]], mfc='r', c=color,
                 lw=.3, ms=5, marker='.')


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    plt.imshow(np.hstack((im1, im2)), cmap="gray")
    inliers1, inliers2 = points1[inliers], points2[inliers]
    plot_marks(points1, points2, im1.shape[1], 'b')
    plot_marks(inliers1, inliers2, im1.shape[1], 'y')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    len_ = len(H_succesive)
    H_result = np.empty((len_ + 1, 3, 3))
    H_result[m] = np.eye(3)
    for i in range(len_):
        if i >= m:
            H = np.dot(H_result[i], np.linalg.inv(H_succesive[i]))
            H_result[i + 1] = H / H[2, 2]
        if i < m:
            H = np.dot(H_result[m - i], H_succesive[m - i - 1])
            H_result[m - i - 1] = H / H[2, 2]
    return list(H_result)


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    h_top_left = apply_homography(np.array([[0, 0]]), homography)
    h_top_right = apply_homography(np.array([[w - 1, 0]]), homography)
    h_bottom_right = apply_homography(np.array([[w - 1, h - 1]]), homography)
    h_bottom_left = apply_homography(np.array([[0, h - 1]]), homography)
    x = [h_top_left[0][0], h_top_right[0][0], h_bottom_right[0][0],
         h_bottom_left[0][0]]
    y = [h_top_left[0][1], h_top_right[0][1], h_bottom_right[0][1],
         h_bottom_left[0][1]]
    min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h, w = image.shape
    [min_x, min_y], [max_x, max_y] = compute_bounding_box(homography, w, h)
    new_coordinates, shape = indexes(np.arange(min_x, max_x + 1),
                                     np.arange(min_y, max_y + 1), homography)
    return map_coordinates(image, new_coordinates, order=1,
                           prefilter=False).reshape(shape)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in
         range(3)])


def filter_homographies_with_translation(homographies,
                                         minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (
            corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [
            os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i
            in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs,
                                                           (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i],
                                                          self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2,
                                    endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros(
            (number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None,
                              :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in
                              self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :,
                                      0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(
            np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:,
                             :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) *
                                      panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros(
            (number_of_panoramas, panorama_size[1], panorama_size[0], 3),
            dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:,
                              boundaries[0] - x_offset: boundaries[
                                                            1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom,
                boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        The bonus
        :param number_of_panoramas: how many different slices to take from each input image
        """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

