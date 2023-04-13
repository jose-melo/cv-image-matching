import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cv_image_matching.feature_extraction.sift import SIFT
from cv_image_matching.utils.profiler import profile

DATABASE = "notre_dame_front_facade"
SRC = "data/train"
IMG1_ID = "01516300_11234314903"
IMG2_ID = "01569849_8047248507"
# IMG1_PATH = "data/index.png"
# IMG2_PATH = "data/index.png"


IMAGE_SIZE = (500, 500)


def get_images():
    img1_path = SRC + "/" + DATABASE + "/images/" + IMG1_ID + ".jpg"
    img2_path = SRC + "/" + DATABASE + "/images/" + IMG2_ID + ".jpg"
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    gray1 = cv.cvtColor(cv.imread(img1_path), cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(cv.imread(img2_path), cv.COLOR_BGR2GRAY)

    gray1 = cv.resize(gray1, IMAGE_SIZE)
    gray2 = cv.resize(gray2, IMAGE_SIZE)

    resized_img1 = cv.resize(img1, dsize=IMAGE_SIZE)
    resized_img2 = cv.resize(img2, dsize=IMAGE_SIZE)

    return img1, img2, gray1, gray2, resized_img1, resized_img2


def drawlines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


@profile(sort_by="cumulative", lines_to_print=50, strip_dirs=True)
def main():
    img1, img2, gray1, gray2, resized_img1, resized_img2 = get_images()

    _, ax = plt.subplots(2, 2)
    ax[0][0].imshow(img1)
    ax[0][1].imshow(img2)
    ax[1][0].imshow(resized_img1)
    ax[1][1].imshow(resized_img2)
    plt.show()

    params = {
        "kp_find_threshold": 1,
        "kp_max_tolerance": 0,
        "local_max_threshold": 10,
        "initial_sigma": 1.6,
        "n_scales_per_octave": 3,
        "n_octaves": 8,
        "assumed_blur": 0.5,
        "gaussian_window_histogram": 1.5,
        "num_bins_histogram": 36,
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

    sift1 = SIFT(**params)
    sift2 = SIFT(**params)
    sift_opencv = cv.SIFT_create()
    kp1_cv, des1_cv = sift_opencv.detectAndCompute(resized_img1, None)
    kp2_cv, des2_cv = sift_opencv.detectAndCompute(resized_img2, None)

    start = time.time()
    kp1, des1 = sift1.detect_and_compute(gray1.astype(np.float32))
    end = time.time()
    print("Calculated keypoints and descriptors for image 1")
    print("Time taken for SIFT: ", end - start, " seconds")
    kp2, des2 = sift2.detect_and_compute(gray2.astype(np.float32))
    start = time.time()
    print("Calculated keypoints and descriptors for image 2")
    end = time.time()
    print("Time taken for SIFT: ", end - start, " seconds")

    des1 = np.array(des1).astype(np.float32)
    des2 = np.array(des2).astype(np.float32)

    _, ax = plt.subplots(2, 2)
    sift_img1 = resized_img1.copy()
    cv.drawKeypoints(gray1, kp1, sift_img1)
    sift_img2 = resized_img2.copy()
    cv.drawKeypoints(gray2, kp2, sift_img2)
    sift_img1_cv = resized_img1.copy()
    cv.drawKeypoints(gray1, kp1_cv, sift_img1_cv)
    sift_img2_cv = resized_img2.copy()
    cv.drawKeypoints(gray2, kp2_cv, sift_img2_cv)

    ax[0][0].imshow(sift_img1, cmap="gray")
    ax[0][1].imshow(sift_img2, cmap="gray")
    ax[1][0].imshow(sift_img1_cv, cmap="gray")
    ax[1][1].imshow(sift_img2_cv, cmap="gray")

    plt.show()

    index_params = {"algorithm": 1, "trees": 5}
    search_params = {"checks": 50}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matches_cv = flann.knnMatch(des1_cv, des2_cv, k=2)

    img1_matched, img2_matched = get_images_matching(
        resized_img1,
        resized_img2,
        kp1,
        kp2,
        matches,
    )
    img1_matched_cv, img2_matched_cv = get_images_matching(
        resized_img1,
        resized_img2,
        kp1_cv,
        kp2_cv,
        matches_cv,
    )

    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].imshow(img1_matched)
    ax[0][1].imshow(img2_matched)
    ax[1][0].imshow(img1_matched_cv)
    ax[1][1].imshow(img2_matched_cv)
    plt.show()


def get_images_matching(img1, img2, kp1, kp2, matches):
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fund_matrix, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print("Fundamental matrix: ", fund_matrix)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fund_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1.copy(), img2.copy(), lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fund_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2.copy(), img1.copy(), lines2, pts2, pts1)
    return img5, img3


if __name__ == "__main__":
    data_path = SRC + "/" + DATABASE + "/pair_covisibility.csv"
    data = pd.read_csv(data_path)
    F_true = data.iloc[0]["fundamental_matrix"]
    F_true = [float(i) for i in F_true.split(" ")]
    F_true = [a / F_true[-1] for a in F_true]
    F_true = np.array(F_true).reshape(3, 3)
    print(F_true)
    main()
