from time import time

import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray
from cv_image_matching.sift.pysift import convertKeypointsToInputImageSize

from cv_image_matching.sift.sift import SIFT

IMG_PATH = "data/train/notre_dame_front_facade/images/01516300_11234314903.jpg"
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
    img = sift.generate_base_image(img, 1.6, 0.5)

    gaussianed_images = sift.gaussian_images(img)

    start = time()
    dog_images = sift.compute_dog_images(gaussianed_images)

    key_points_cv, key_points = sift.find_keypoints(
        dog_images,
        threshold=1e-3,
        max_tolerance=1,
    )
    print(f"Total keypoints: {len(key_points)}")
    filtered_key_points = sift.filter_keypoints(dog_images, key_points, threshold=10)
    print(f"Filtered keypoints: {len(filtered_key_points)}")
    end = time()
    print(f"Time: {end - start} seconds")

    scaled_keypoints = sift.convert_keypoints(filtered_key_points)

    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    wwimg = cv.drawKeypoints(gray, scaled_keypoints, img)

    # plt.show()
    sift = cv.SIFT_create()
    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
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
