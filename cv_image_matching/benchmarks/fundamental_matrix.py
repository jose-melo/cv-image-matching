import time

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from kornia.feature import LoFTR
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
from models.matching import Matching


@profile(sort_by="cumulative", lines_to_print=50, strip_dirs=True)
def find_fund_matrix(
    img1_path: str,
    img2_path: str,
    img_size: tuple[int, int],
    experiment: str = "own",
    params: dict = None,
    show: bool = False,
) -> dict[str, ndarray]:
    """Find fundamental matrix using own implementation and opencv implementation

    Args:
        img1_path (str): Path to image 1
        img2_path (str): Path to image 2
        img_size (tuple[int, int]): Size of the images
        params (dict): Parameters for SIFT
        show (bool, optional): Wheter or not to plot the images. Defaults to False.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
            f_own, f_cv, inlier_kp1, inlier_kp2, inlier_kp1_cv, inlier_kp2_cv
    """

    gray1, gray2, resized_img1, resized_img2 = get_images(
        img1_path,
        img2_path,
        img_size,
        show,
    )

    # Own SIFT
    if experiment == "own_sift":
        kp1, des1, kp2, des2 = run_feature_extracion_own(gray1, gray2, params)
        matches = get_matches(des1, des2)
        f, kp1, kp2 = get_fund_matrix_from_matches(kp1, kp2, matches)

    # OpenCV SIFT
    if experiment == "opencv_sift":
        sift_opencv = cv.SIFT_create()
        kp1, des1 = sift_opencv.detectAndCompute(resized_img1.copy(), None)
        kp2, des2 = sift_opencv.detectAndCompute(resized_img2.copy(), None)
        matches = get_matches(des1, des2)
        f, kp1, kp2 = get_fund_matrix_from_matches(kp1, kp2, matches)

    # SuperGlue
    if experiment == "superglue":
        f, kp1, kp2 = run_feature_extraction_glue(gray1, gray2)

    if experiment == "loftr":
        f, kp1, kp2 = run_feature_extraction_loftr(gray1, gray2)

    if show:
        plot_keypoints(gray1, gray2, resized_img1, resized_img2, kp1, kp2)

    return f, kp1, kp2


def run_feature_extraction_loftr(gray1, gray2):
    img1 = torch.from_numpy(gray1)[None][None] / 255.0
    img2 = torch.from_numpy(gray2)[None][None] / 255.0

    matcher = LoFTR(pretrained="outdoor")

    input_dict = {
        "image0": img1,
        "image1": img2,
    }

    with torch.inference_mode():
        matches = matcher(input_dict)

    mkpts0 = matches["keypoints0"].cpu().numpy()
    mkpts1 = matches["keypoints1"].cpu().numpy()
    f_loftr, inliers = cv.findFundamentalMat(
        mkpts0,
        mkpts1,
        cv.USAC_MAGSAC,
        0.5,
        0.999,
        100000,
    )
    inliers = inliers > 0

    return f_loftr, mkpts0, mkpts1


def run_feature_extraction_glue(gray1, gray2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "superpoint": {
            "nms_radius": 3,
            "keypoint_threshold": 0.005,
            "max_keypoints": 2048,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 100,
            "match_threshold": 0.2,
        },
    }
    matching = Matching(config).eval().to(device)
    img1_glue = torch.from_numpy(gray1)[None][None] / 255.0
    img2_glue = torch.from_numpy(gray2)[None][None] / 255.0

    start = time.time()
    pred = matching({"image0": img1_glue, "image1": img2_glue})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    end = time.time()
    print("Total time: {:.2f} ms".format((end - start) * 1000))

    matches = pred["matches0"]
    kp1, kp2 = pred["keypoints0"], pred["keypoints1"]
    valid = matches > -1
    mkpts1 = kp1[valid]
    mkpts2 = kp2[matches[valid]]

    f_superglue, inlier_mask = cv.findFundamentalMat(
        mkpts1,
        mkpts2,
        cv.USAC_MAGSAC,
        ransacReprojThreshold=0.25,
        confidence=0.99999,
        maxIters=10000,
    )

    return f_superglue, mkpts1, mkpts2


def run_evaluation(
    folder: str,
    src: str,
    params: dict,
    experiments: list[str] = ["own_sift", "opencv_sift", "superglue"],
    idx: int = -1,
    img_size: tuple[int, int] = (200, 200),
    show: bool = False,
) -> dict:
    """Run evaluation on a given scene and index.

    Args:
        folder (str): Path to the scene folder
        idx (int, optional): Index of the case to be treated. Defaults to -1.
        show (bool, optional): Wheter or not to plot images. Defaults to False.
        img_size (tuple[int, int], optional): Size of the images. Defaults to (200, 200)
        src (str, optional): Source path of the data.
    Returns:
        tuple[float, float, float, float]:
            The errors (translation and rotation)
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

    errors = get_errors_for_case(
        experiments,
        params,
        folder,
        idx,
        img1_path,
        img2_path,
        img_size,
        src,
        img1_id,
        img2_id,
        data,
        show,
    )

    return errors


def get_errors_for_case(
    experiments: list[str],
    params: dict,
    folder: str,
    idx: int,
    img1_path: str,
    img2_path: str,
    img_size: tuple[int, int],
    src: str,
    img1_id: str,
    img2_id: str,
    data: pd.DataFrame,
    show: bool = False,
) -> dict:
    """Calculates the fundamental matrices and generate the errors for the given case.

    Args:
        img1_path (str): Path of the first image.
        img2_path (str): Path of the second image.
        img_size (tuple[int, int]): Size of the images.
        src (str): Source path of the data.
        img1_id (str): Id of the first image.
        img2_id (str): Id of the second image.
        data (pd.DataFrame): Dataframe with the data.
        show (bool, optional): Wheter or not to plot images. Defaults to False.

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

    general_config_fund_matrix = {
        "img1_path": img1_path,
        "img2_path": img2_path,
        "img_size": img_size,
        "show": show,
    }

    general_params_errors = {
        "K1": k1,
        "K2": k2,
        "R1": r1,
        "R2": r2,
        "T1": t1,
        "T2": t2,
        "scaling_factor": scale,
    }

    f_true = np.array(
        [float(x) for x in data.iloc[idx]["fundamental_matrix"].split(" ")],
    )

    errors = {exp: {} for exp in experiments}

    for exp in experiments:
        print(f"Starting experiment: {exp}")
        f, kp1, kp2 = find_fund_matrix(
            **general_config_fund_matrix,
            experiment=exp,
            params=params,
        )

        print("Computing errors...")
        err_q, err_t = compute_errors(
            F=f,
            kp1=kp1,
            kp2=kp2,
            **general_params_errors,
        )
        err_f = np.linalg.norm(f.ravel() - f_true)

        errors[exp]["err_q"] = err_q
        errors[exp]["err_t"] = err_t
        errors[exp]["err_f"] = err_f

    return errors
