from time import time

import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray
from sift import SIFT

IMG_PATH = "data/train/notre_dame_front_facade/images/01516300_11234314903.jpg"
IMG_PATH = "data/index.png"


def load_image(path: str) -> ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img


def main():
    img = load_image(IMG_PATH)
    img = img.astype("float32")

    params = {
        "initial_sigma": 1.6,
        "n_scales_per_octave": 3,
        "n_octaves": 8,
        "assumed_blur": 0.5,
        "kp_find_threshold": 1e-3,
        "kp_max_tolerance": 1,
        "local_max_threshold": 10,
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
    sift = SIFT(**params)
    img = sift.generate_base_image(img)

    sift.gaussian_images()

    start = time()
    sift.compute_dog_images()

    sift.find_keypoints()
    print(f"Total keypoints: {len(sift.keypoints)}")
    sift.filter_keypoints()
    print(f"Filtered keypoints: {len(sift.filtered_keypoints)}")
    end = time()
    print(f"Time: {end - start} seconds")

    sift.convert_keypoints()

    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    wwimg = cv.drawKeypoints(gray, sift.scaled_keypoints, img)

    sift = cv.SIFT_create()
    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    print("Number of keypoints OpenCV SIFT:", len(kp))
    img = cv.drawKeypoints(gray, kp, img)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(wwimg, cmap="gray", label="own")
    ax[0].set_title("own")
    ax[1].imshow(img, cmap="gray", label="opencv")
    ax[1].set_title("opencv")
    plt.show()


if __name__ == "__main__":
    main()
