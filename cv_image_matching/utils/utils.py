import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray

from cv_image_matching.feature_extraction.sift import SIFT


def get_images(
    img1_path: str,
    img2_path: str,
    image_size: tuple[int, int] = (200, 200),
    show: bool = False,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Get images and resize them

    Args:
        img1_path (str): Path to image 1
        img2_path (str): Path to image 2
        image_size (tuple, optional): Final size of the images. Defaults to IMAGE_SIZE.
        show (bool, optional): Wheter or not to plot the images. Defaults to False.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]:
            gray1, gray2, resized_img1, resized_img2
    """
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    gray1 = cv.cvtColor(cv.imread(img1_path), cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(cv.imread(img2_path), cv.COLOR_BGR2GRAY)

    gray1 = cv.resize(gray1, image_size)
    gray2 = cv.resize(gray2, image_size)

    resized_img1 = cv.resize(img1, dsize=image_size)
    resized_img2 = cv.resize(img2, dsize=image_size)

    if show:
        _, ax = plt.subplots(2, 2)
        ax[0][0].imshow(img1)
        ax[0][1].imshow(img2)
        ax[1][0].imshow(resized_img1)
        ax[1][1].imshow(resized_img2)
        plt.show()

    return gray1, gray2, resized_img1, resized_img2


def get_camera_calibration(
    folder: str,
    src: str,
    img1_id: str,
    img2_id: str,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, float]:
    """Read dataset calibration data. Returns the rotation and translation matrices, the
    intrinsics matrix R1, R2, T1, T2, K1, K2 and the scaling factor.

    Args:
        folder (str): folder path where to read the csv.
        src (str): path to the datasets
        img1_id (str): id of the first image
        img2_id (str): id of the second image

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, float]:
            rotation and translation matrices, intrinsics matrix, scaling factor
    """

    intrinsics_path = src + "/" + folder + "/calibration.csv"
    intrinsics = pd.read_csv(intrinsics_path)
    intrinsics = intrinsics.set_index("image_id", drop=True)
    scaling_path = src + "/" + "/scaling_factors.csv"
    scales = pd.read_csv(scaling_path)
    scales = scales.set_index("scene", drop=True)

    r1 = read_intrinsics_matrix(img1_id, intrinsics, "rotation_matrix").reshape(3, 3)

    r2 = read_intrinsics_matrix(img2_id, intrinsics, "rotation_matrix").reshape(3, 3)

    t1 = read_intrinsics_matrix(img1_id, intrinsics, "translation_vector").reshape(3, 1)

    t2 = read_intrinsics_matrix(img2_id, intrinsics, "translation_vector").reshape(3, 1)

    k1 = read_intrinsics_matrix(img1_id, intrinsics, "camera_intrinsics").reshape(3, 3)

    k2 = read_intrinsics_matrix(img2_id, intrinsics, "camera_intrinsics").reshape(3, 3)

    scale = scales.loc[folder]["scaling_factor"]

    return r1, r2, t1, t2, k1, k2, scale


def read_intrinsics_matrix(img1_id: str, intrinsics: pd.DataFrame, matrix: str):
    return np.array(
        [float(x) for x in intrinsics.loc[img1_id][matrix].split(" ")],
    )


def plot_keypoints(
    gray1: ndarray,
    gray2: ndarray,
    resized_img1: ndarray,
    resized_img2: ndarray,
    kp1: ndarray,
    kp2: ndarray,
    experiment: str = "Own SIFT",
):
    """Plot the keypoints of the two images

    Args:
        gray1 (ndarray): Image 1 in grayscale
        gray2 (ndarray): Image 2 in grayscale
        resized_img1 (ndarray): Resized image 1
        resized_img2 (ndarray): Resized image 2
        kp1 (ndarray): Key points of image 1
        kp2 (ndarray): Key points of image 2
        kp1_cv (ndarray): Key points of image 1 using OpenCV
        kp2_cv (ndarray): Key points of image 2 using OpenCV
    """
    _, ax = plt.subplots(1, 2)
    sift_img1 = resized_img1.copy()
    cv.drawKeypoints(gray1, kp1, sift_img1)
    sift_img2 = resized_img2.copy()
    cv.drawKeypoints(gray2, kp2, sift_img2)

    ax[0].imshow(sift_img1, cmap="gray")
    ax[0].set_title(f"{experiment} keypoints Image 1")
    ax[1].imshow(sift_img2, cmap="gray")
    ax[1].set_title(f"{experiment} keypoints Image 2")
    plt.show()


def get_fund_matrix_from_matches(
    kp1: list[cv.KeyPoint],
    kp2: list[cv.KeyPoint],
    matches: list[cv.DMatch],
) -> tuple[ndarray, ndarray, ndarray]:
    """Get the fundamental matrix from the matches

    Args:
        kp1 (list[cv.KeyPoint]): Key points of image 1
        kp2 (list[cv.KeyPoint]): Key points of image 2
        matches (list[cv.DMatch]): Matches between the two images

    Returns:
        tuple[ndarray, ndarray, ndarray]:
            f, kp1, kp2
    """
    matches_own = np.array([[m[0].queryIdx, m[0].trainIdx] for m in matches])
    cur_kp_1_own = array_from_cv_kps([kp1[m[0]] for m in matches_own])
    cur_kp_2_own = array_from_cv_kps([kp2[m[1]] for m in matches_own])

    f_own, inlier_mask = cv.findFundamentalMat(
        cur_kp_1_own,
        cur_kp_2_own,
        cv.USAC_MAGSAC,
        0.25,
        0.99999,
        10000,
    )
    inlier_mask = inlier_mask.astype(bool).flatten()

    matches_after_ransac = np.array(
        [match for match, is_inlier in zip(matches_own, inlier_mask) if is_inlier],
    )
    inlier_kp1 = array_from_cv_kps([kp1[m[0]] for m in matches_after_ransac])
    inlier_kp2 = array_from_cv_kps([kp2[m[1]] for m in matches_after_ransac])
    return f_own, inlier_kp1, inlier_kp2


def get_matches(des1: ndarray, des2: ndarray) -> tuple[cv.FlannBasedMatcher, list]:
    """Match keypoints between two images using the FLANN algorithm

    Args:
        des1 (ndarray): array of descriptors for the first image
        des2 (ndarray): array of descriptors for the second image

    Returns:
        tuple[cv.FlannBasedMatcher, list]: FLANN matcher and matches
    """

    index_params = {"algorithm": 1, "trees": 5}
    search_params = {"checks": 50}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(
        des1,
        des2,
        k=2,
    )

    return matches


def run_feature_extracion_own(
    gray1: ndarray,
    gray2: ndarray,
    params: dict,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Run feature extraction on two images

    Args:
        gray1 (ndarray): First image in grayscale
        gray2 (ndarray): Second image in grayscale
        params (dict): Parameters for the SIFT algorithm
        show (bool, optional): Wheter or not to show images. Defaults to False.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
            keypoints and descriptors for both images
    """

    sift1 = SIFT(**params)
    sift2 = SIFT(**params)

    start = time.time()
    kp1, des1 = sift1.detect_and_compute(gray1.astype(np.float32))
    end = time.time()
    print("Calculated keypoints and descriptors for image 1")
    print("Time taken for own SIFT: ", end - start, " seconds")
    start = time.time()
    kp2, des2 = sift2.detect_and_compute(gray2.astype(np.float32))
    print("Calculated keypoints and descriptors for image 2")
    end = time.time()
    print("Time taken for OpenCV SIFT: ", end - start, " seconds")

    return kp1, des1, kp2, des2


# --------------------- #
# The following code is taken from the following tutorial:
# https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
# --------------------- #


def drawlines(
    img1: ndarray,
    img2: ndarray,
    lines: ndarray,
    pts1: ndarray,
    pts2: ndarray,
) -> tuple[ndarray, ndarray]:
    """Draw epilines on the images

    Args:
        img1 (ndarray): Image 1
        img2 (ndarray): IMage 2
        lines (ndarray): Lines to draw
        pts1 (ndarray): Points in image 1
        pts2 (ndarray): POints in image 2

    Returns:
        tuple[ndarray, ndarray]: img1, img2
    """
    r, c, _ = img1.shape
    generator = np.random.default_rng()
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(generator.integers(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def get_images_matching(
    img1: ndarray,
    img2: ndarray,
    kp1: ndarray,
    kp2: ndarray,
    matches: ndarray,
) -> tuple[ndarray, ndarray]:
    """Get the images with the matching points

    Args:
        img1 (ndarray): Image 1
        img2 (ndarray): Image 2
        kp1 (ndarray): Keypoints of image 1
        kp2 (ndarray): Keypoints of image 2
        matches (ndarray): Matches between the keypoints

    Returns:
        tuple[ndarray, ndarray]:
            img1_with_matches, img2_with_matches
    """
    pts1 = []
    pts2 = []

    ratio = 0.7
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fund_matrix, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fund_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, _ = drawlines(img1.copy(), img2.copy(), lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fund_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, _ = drawlines(img2.copy(), img1.copy(), lines2, pts2, pts1)
    return img5, img3, fund_matrix


# ----------------------------------- #
# The following code is inspired by:
# https://www.kaggle.com/competitions/image-matching-challenge-2022/overview/evaluation
# ----------------------------------- #


def compute_maa(
    err_q: ndarray,
    err_t: ndarray,
    thresholds_q: ndarray,
    thresholds_t: ndarray,
) -> tuple[float, ndarray, ndarray, ndarray]:
    """Compute the mean Average Accuracy at different tresholds, for one scene."""

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [
            (np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum()
            / len(err_q),
        ]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)


def normalize_keypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints


def compute_essencial_matrix(F, K1, K2, kp1, kp2):
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)

    kp1n = normalize_keypoints(kp1, K1)
    kp2n = normalize_keypoints(kp2, K2)
    _, R, T, _ = cv.recoverPose(E, kp1n, kp2n)

    return E, R, T


def quaternion_from_matrix(matrix):
    """Transform a rotation matrix into a quaternion."""

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def compute_error_for_one_example(q_gt, T_gt, q, T, scale):
    """Compute the error metric for a single example.

    The function returns two errors, over rotation and translation.
    These are combined at different thresholds by ComputeMaa in order
    to compute the mean Average Accuracy.
    """

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + 1e-8)
    q_norm = q / (np.linalg.norm(q) + 1e-8)

    loss_q = np.maximum(1e-8, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + 1e-8)

    err_t = min(
        np.linalg.norm(T_gt_scaled - T_scaled),
        np.linalg.norm(T_gt_scaled + T_scaled),
    )

    return err_q * 180 / np.pi, err_t


def compute_errors(F, K1, K2, R1, R2, T1, T2, kp1, kp2, scaling_factor):
    # Compute matches by brute force.
    # Compute the essential matrix.

    E, R, T = compute_essencial_matrix(F, K1, K2, kp1, kp2)
    q = quaternion_from_matrix(R)
    T = T.flatten()

    R1_gt, T1_gt = R1, T1.reshape((3, 1))
    R2_gt, T2_gt = R2, T2.reshape((3, 1))

    dR_gt = np.dot(R2_gt, R1_gt.T)
    dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()

    q_gt = quaternion_from_matrix(dR_gt)
    q_gt = q_gt / (np.linalg.norm(q_gt) + 1e-8)

    # Compute the error for this example.
    err_q, err_t = compute_error_for_one_example(q_gt, dT_gt, q, T, scaling_factor)

    return err_q, err_t


def array_from_cv_kps(kps):
    """Convenience function to convert OpenCV keypoints into a simple numpy array."""
    return np.array([kp.pt for kp in kps])
