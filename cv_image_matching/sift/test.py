import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray
from time import time
from cv_image_matching.sift.pysift import (
    generateBaseImage,
    generateDoGImages,
    generateGaussianImages,
    generateGaussianKernels,
)

from cv_image_matching.sift.sift import SIFT


IMG_PATH = "data/index.png"


def load_image(path: str) -> ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # print("Image shape:", img.shape)
    # plt.imshow(img, cmap="gray")
    # plt.show()
    return img


def main():
    img = load_image(IMG_PATH)
    img = img.astype("float32")

    params = {"initial_sigma": 1.6, "n_scales_per_octave": 3, "n_octaves": 8}
    sift = SIFT(**params)
    img = generateBaseImage(img, 1.6, 0.5)

    gaussianed_images = sift.gaussian_images(img)
    # gaussian_kernels = generateGaussianKernels(1.6, 3)
    # gaussian_kernels_2 = sift.compute_gaussian_scales()
    # print(gaussian_kernels)
    # print(gaussian_kernels_2)

    # fig, ax = plt.subplots(sift.n_octaves, sift.n_intervals, figsize=(20, 10))
    # for i, octave in gaussianed_images.items():
    # for j, img_scale in octave.items():
    # ax[i][j].imshow(img_scale, cmap="gray")
    # plt.show()

    start = time()
    dog_images = sift.compute_dog_images(gaussianed_images)
    fig, ax = plt.subplots(len(dog_images), len(dog_images[0]), figsize=(20, 10))
    for i, octave in dog_images.items():
        for j, img_scale in octave.items():
            ax[i][j].imshow(img_scale, cmap="gray")
    plt.show()

    key_points = sift.find_keypoints(dog_images)
    print(f"Total keypoints: {len(key_points)}")
    return
    filtered_key_points = sift.filter_keypoints(dog_images, key_points, threshold=1)
    print(f"Filtered keypoints: {len(filtered_key_points)}")
    end = time()
    print(f"Time: {end - start} seconds")

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for idx, point_list in enumerate(
        [
            (key_points, "Total keypoints"),
            (filtered_key_points, "Filtered keypoints, sigma_max = 5"),
        ]
    ):
        point_list_data = np.array(point_list[0])[:, 2:]
        _img1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        for i, j in point_list_data:
            _img1[i][j][0] = 255
            _img1[i][j][1] = 0
            _img1[i][j][2] = 0
        ax[idx].imshow(_img1)
        ax[idx].set_title(point_list[1])

    plt.show()
    sift = cv.SIFT_create()
    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    img = cv.drawKeypoints(gray, kp, img)
    plt.imshow(img, cmap="gray")
    plt.show()

    # hist = sift.calculate_dominant_histogram(dog_images, 1, 2, 2, 2)


if __name__ == "__main__":
    main()
