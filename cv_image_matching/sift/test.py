from time import time

import cv2 as cv
import matplotlib.pyplot as plt
from numpy import ndarray

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

    params = {"initial_sigma": 1.6, "n_scales_per_octave": 10, "n_octaves": 8}
    sift = SIFT(**params)
    img = sift.generate_base_image(img, 1.6, 0.5)

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
    # fig, ax = plt.subplots(len(dog_images), len(dog_images[0]), figsize=(20, 10))
    # for i, octave in dog_images.items():
    # for j, img_scale in octave.items():
    # ax[i][j].imshow(img_scale, cmap="gray")
    # plt.show()

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

    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # for idx, point_list in enumerate(
    # [
    # (key_points_cv, "Total keypoints"),
    # (filtered_key_points, "Filtered keypoints, sigma_max = 5"),
    # ]
    # ):
    ## print(point_list, np.array(point_list[0]).shape, point_list[0])
    # point_list_data = np.array(point_list[0])
    # _img1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # pts = np.asarray([[p.pt[0], p.pt[1]] for p in point_list_data])
    # cols = pts[:, 0]
    # rows = pts[:, 1]
    # ax[idx].imshow(img)
    # ax[idx].scatter(cols, rows)
    # ax[idx].set_title(point_list[1])

    # plt.show()
    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    wwimg = cv.drawKeypoints(gray, key_points_cv, img)
    plt.imshow(wwimg, cmap="gray")
    plt.show()

    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    wwimg = cv.drawKeypoints(gray, filtered_key_points, img)
    plt.imshow(wwimg, cmap="gray")
    plt.show()
    # for idx, point_list in enumerate(
    # [
    # (key_points, "Total keypoints"),
    # (filtered_key_points, "Filtered keypoints, sigma_max = 5"),
    # ]
    # ):
    # point_list_data = np.array(point_list[0])[:, 2:]
    # _img1 = load_image(IMG_PATH)
    # _img1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # for i, j in point_list_data:
    # _img1[i][j][0] = 255
    # _img1[i][j][1] = 0
    # _img1[i][j][2] = 0
    # ax[idx].imshow(_img1, cmap="gray")
    # ax[idx].set_title(point_list[1])

    # plt.show()
    sift = cv.SIFT_create()
    gray = cv.cvtColor(cv.imread(IMG_PATH), cv.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    print("Number of keypoints OpenCV SIFT:", len(kp))
    img = cv.drawKeypoints(gray, kp, img)
    plt.imshow(img, cmap="gray")
    plt.show()

    # hist = sift.calculate_dominant_histogram(dog_images, 1, 2, 2, 2)


if __name__ == "__main__":
    main()
