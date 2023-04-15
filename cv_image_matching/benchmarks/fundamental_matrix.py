import cv2 as cv
import numpy as np
import pandas as pd
from numpy import ndarray

from cv_image_matching.utils.profiler import profile
from cv_image_matching.utils.utils import (
    compute_errors,
    get_camera_calibration,
    get_fund_matrix_from_matches,
    get_images,
    get_matches,
    plot_keypoints,
    run_feature_extracion_own,
)


@profile(sort_by="cumulative", lines_to_print=50, strip_dirs=True)
def find_fund_matrix(
    img1_path: str,
    img2_path: str,
    params: dict,
    show: bool = False,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Find fundamental matrix using own implementation and opencv implementation

    Args:
        img1_path (str): Path to image 1
        img2_path (str): Path to image 2
        params (dict): Parameters for SIFT
        show (bool, optional): Wheter or not to plot the images. Defaults to False.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
            f_own, f_cv, inlier_kp1, inlier_kp2, inlier_kp1_cv, inlier_kp2_cv
    """

    gray1, gray2, resized_img1, resized_img2 = get_images(
        img1_path,
        img2_path,
        show,
    )

    kp1, des1, kp2, des2 = run_feature_extracion_own(gray1, gray2, params)

    sift_opencv = cv.SIFT_create()
    kp1_cv, des1_cv = sift_opencv.detectAndCompute(resized_img1.copy(), None)
    kp2_cv, des2_cv = sift_opencv.detectAndCompute(resized_img2.copy(), None)

    if show:
        plot_keypoints(
            gray1,
            gray2,
            resized_img1,
            resized_img2,
            kp1,
            kp2,
            kp1_cv,
            kp2_cv,
        )

    matches = get_matches(des1, des2)
    matches_cv = get_matches(des1_cv, des2_cv)

    f_own, kp1_own, kp2_own = get_fund_matrix_from_matches(kp1, kp2, matches)
    f_cv, kp1_cv, kp2_cv = get_fund_matrix_from_matches(
        kp1_cv,
        kp2_cv,
        matches_cv,
    )

    return f_own, f_cv, kp1_own, kp2_own, kp1_cv, kp2_cv


def run_evaluation(
    folder: str,
    src: str,
    params: dict,
    idx: int = -1,
    show: bool = False,
):
    """Run evaluation on a given scene and index.

    Args:
        folder (str): Path to the scene folder
        idx (int, optional): Index of the case to be treated. Defaults to -1.
        show (bool, optional): Wheter or not to plot images. Defaults to False.
        src (str, optional): Source path of the data.
    """
    data_path = src + "/" + folder + "/pair_covisibility.csv"
    data = pd.read_csv(data_path)

    generator = np.random.default_rng(42)
    if idx < 0:
        idx = generator.integers(len(data))

    img1_id = data.iloc[idx]["pair"].split("-")[0]
    img2_id = data.iloc[idx]["pair"].split("-")[1]

    img1_path = src + "/" + folder + "/images/" + img1_id + ".jpg"
    img2_path = src + "/" + folder + "/images/" + img2_id + ".jpg"

    err_q_own, err_t_own, err_q_opencv, err_t_opencv = get_errors_for_case(
        params,
        img1_path,
        img2_path,
        src,
        img1_id,
        img2_id,
        show,
    )

    print("Own: ", err_q_own, err_t_own)
    print("OpenCV: ", err_q_opencv, err_t_opencv)


def get_errors_for_case(
    params: dict,
    img1_path: str,
    img2_path: str,
    src: str,
    img1_id: str,
    img2_id: str,
    show: bool = False,
) -> tuple[float, float, float, float]:
    """Calculates the fundamental matrices and generate the errors for the given case.

    Args:
        img1_path (str): Path of the first image.
        img2_path (str): Path of the second image.
        r1 (ndarray): Rotation matrix of the first camera.
        r2 (ndarray): Rotation matrix of the second camera.
        t1 (ndarray): Translation matrix of the first camera.
        t2 (ndarray): Translation matrix of the second camera.
        k1 (ndarray): Intrinsics matrix of the first camera.
        k2 (ndarray): Intrinsics matrix of the second camera.
        scale (float): Scaling factor of the scene.

    Returns:
        tuple[float, float, float, float]:
            The errors (translation and rotation) for the own implementation
            and the OpenCV implementation.
    """
    r1, r2, t1, t2, k1, k2, scale = get_camera_calibration(
        folder,
        src,
        img1_id,
        img2_id,
    )

    f_own, f_opencv, kp1_own, kp2_own, kp1_cv, kp2_cv = find_fund_matrix(
        img1_path,
        img2_path,
        params,
        show,
    )

    err_q_own, err_t_own = compute_errors(
        f_own,
        k1,
        k2,
        r1,
        r2,
        t1,
        t2,
        kp1_own,
        kp2_own,
        scale,
    )
    err_q_opencv, err_t_opencv = compute_errors(
        f_opencv,
        k1,
        k2,
        r1,
        r2,
        t1,
        t2,
        kp1_cv,
        kp2_cv,
        scale,
    )

    return err_q_own, err_t_own, err_q_opencv, err_t_opencv


if __name__ == "__main__":
    folder = "notre_dame_front_facade"
    src = "data/train"
    image_size = (200, 200)
    idx = 0

    params_sift = {
        "kp_find_threshold": 1,
        "kp_max_tolerance": 0,
        "local_max_threshold": 10,
        "initial_sigma": 1.6,
        "n_scales_per_octave": 3,
        "n_octaves": 8,
        "assumed_blur": 0.5,
        "gaussian_window_histogram": 1.5,
        "num_bins_histogram": 180,
        "ksize_smooth_histogram": 5,
        "std_smooth_histogram": 1,
        "size_factor": 5,
        "n_spacial_bins": 4,
        "n_orientation_bins": 8,
        "f_max": 0.2,
        "f_scale": 512,
        "descriptor_filter_scale_factor": 0.25,
        "descriptor_cutoff_factor": 2.5,
    }

    run_evaluation(folder, src, params_sift, idx, show=False)
