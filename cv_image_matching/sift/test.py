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
    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)

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
    own_sift = SIFT(**params)
    own_sift.detect_and_compute(img)

    sift_opencv = cv.SIFT_create()
    kp, des = sift_opencv.detectAndCompute(gray, None)

    own_sift_img = cv.drawKeypoints(gray, own_sift.scaled_keypoints, img)
    opencv_img = cv.drawKeypoints(gray, kp, img)

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(own_sift_img, cmap="gray", label="own")
    ax[0].set_title("Own implementation of SIFT")
    ax[1].imshow(opencv_img, cmap="gray", label="opencv")
    ax[1].set_title("Opencv implementation of SIFT")
    plt.show()


if __name__ == "__main__":
    main()
