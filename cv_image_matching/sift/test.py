import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray

from cv_image_matching.sift.sift import SIFT


IMG_PATH = "data/foto1.jpg"


def load_image(path: str) -> ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap="gray")
    plt.show()
    return img


def main():
    img = load_image(IMG_PATH)

    params = {"initial_sigma": 15, "n_scales_per_octave": 5, "n_octaves": 2}
    sift = SIFT(**params)
    gaussianed_images = sift.gaussian_images(img)

    fig, ax = plt.subplots(sift.n_octaves, sift.n_intervals, figsize=(20, 10))
    for octave in range(sift.n_octaves):
        for scale in range(sift.n_intervals):
            ax[octave][scale].imshow(gaussianed_images[octave][scale], cmap="gray")
    plt.show()

    dog_images = sift.compute_dog_images(gaussianed_images)
    key_points = sift.find_keypoints(dog_images)
    filtered_key_points = sift.filter_keypoints(dog_images, key_points, threshold=1)

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
        ax[idx].imshow(_img1)
        ax[idx].set_title(point_list[1])

    plt.show()


if __name__ == "__main__":
    main()
