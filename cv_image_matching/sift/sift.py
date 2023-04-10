from typing import Tuple
import numpy as np
from cv2 import (
    INTER_LINEAR,
    GaussianBlur,
    KeyPoint,
    filter2D,
    getGaussianKernel,
    resize,
    subtract,
)
from numpy import float32, ndarray
from peaks import interpolate


class SIFT:
    def __init__(
        self,
        initial_sigma: float = 1.6,
        n_scales_per_octave: int = 3,
        n_octaves: int = 4,
    ) -> None:
        self.initial_sigma = initial_sigma
        self.n_scales_per_octave = n_scales_per_octave
        self.n_octaves = n_octaves
        self.keypoins = None
        self.keypoints_cv = None
        self.filtered_keypoints = None

    def compute_gaussian_scales(self) -> ndarray:
        """Create a list of gaussian kernels for each octave and scale"""
        # We must produce s + 3 images in the stack of blurred images for each octave
        self.n_intervals = self.n_scales_per_octave + 3

        self.gaussian_scales = np.zeros(self.n_intervals)
        self.gaussian_scales[0] = self.initial_sigma
        for step in range(1, self.n_intervals):
            self.gaussian_scales[step] = (
                self.initial_sigma
                * (2 ** ((step - 1) / (self.n_intervals - 3)))
                * np.sqrt(2 ** (2 / (self.n_intervals - 3)) - 1)
            )

        return self.gaussian_scales

    def gaussian_images(self, image: np.ndarray) -> ndarray:
        """Create a stack of blurred images for each octave and scale.

        Args:
            image (np.ndarray): Input image.
        """

        if not hasattr(self, "gaussian_scales"):
            self.compute_gaussian_scales()

        gaussianed_images = {}

        for octave in range(self.n_octaves):
            gaussianed_images[octave] = {}

            gaussianed_images[octave][0] = image
            for idx, scale in enumerate(self.gaussian_scales[1:]):
                new_image = GaussianBlur(image, (0, 0), sigmaX=scale, sigmaY=scale)
                gaussianed_images[octave][idx + 1] = new_image

            image = resize(
                gaussianed_images[octave][idx - 2],
                (image.shape[1] // 2, image.shape[0] // 2),
            )

        return gaussianed_images

    def generate_base_image(
        self,
        image: np.ndarray,
        sigma: float,
        assumed_blur: float,
    ) -> np.ndarray:
        """Upsamples the input image by 2 in both directions
        and applies Gaussian blur to generate a base image.

        Args:
            image: The input image to generate the base image from.
            sigma: The desired standard deviation of the Gaussian blur.
            assumed_blur: The assumed standard deviation of the blur in the input image.

        Returns:
            The base image, which is the input image upsampled by 2
            in both directions and blurred with a Gaussian kernel
            with standard deviation `sigma_diff`.

        """
        upscaled_image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = np.sqrt(max(sigma**2 - (2 * assumed_blur) ** 2, 0.01))
        base_image = GaussianBlur(
            upscaled_image,
            (0, 0),
            sigmaX=sigma_diff,
            sigmaY=sigma_diff,
        )

        return base_image

    def compute_dog_images(self, gaussianed_images: ndarray) -> ndarray:
        """Compute the difference of gaussian images for each octave and scale.

        Args:
            gaussian_images (ndarray): Stack of blurred images for each octave and scale
        """
        dog_images = {}
        for octave in range(self.n_octaves):
            dog_images[octave] = {}
            for scale in range(self.n_intervals - 1):
                dog_images[octave][scale] = subtract(
                    gaussianed_images[octave][scale + 1],
                    gaussianed_images[octave][scale],
                )
        return dog_images

    def is_keypoint(
        self,
        dog_images: ndarray,
        octave: int,
        scale: int,
        i: int,
        j: int,
        threshold: float = 0.01,
        max_tolerance: int = 1,
    ) -> bool:
        """Check if a pixel is a keypoint (local maximum)."""

        center_pixel = dog_images[octave][scale][i][j]

        # print(center_pixel, i, j, octave, scale, bound_x, bound_y)
        if abs(center_pixel) < threshold:
            return False

        count_fails = 0
        for x in np.arange(-1, 2):
            for y in np.arange(-1, 2):
                for z in np.arange(-1, 2):
                    # print(dog_images[octave][scale + z][i + x][j + y])
                    bound_x, bound_y = dog_images[octave][scale + z].shape
                    if 0 <= i + x < bound_x and 0 <= j + y < bound_y:
                        if x == 0 and y == 0 and z == 0:  # center pixel
                            continue
                        if (
                            center_pixel > 0
                            and center_pixel
                            < dog_images[octave][scale + z][i + x][j + y]
                        ):
                            count_fails += 1
                            if count_fails > max_tolerance:
                                return False
                        if (
                            center_pixel < 0
                            and center_pixel
                            > dog_images[octave][scale + z][i + x][j + y]
                        ):
                            count_fails += 1
                            if count_fails > max_tolerance:
                                return False
        return True

    def find_keypoints(
        self,
        dog_images: ndarray,
        threshold: float = 0.01,
        max_tolerance: int = 1,
    ) -> ndarray:
        """Find the keypoints in the difference of gaussian images.

        Args:
            dog_images (ndarray): Stack of difference of gaussian images
            for each octave and scale.
        """
        keypoints_cv = []
        keypoints = []
        # print("n_octaves: ", self.n_octaves)
        # print("n_scales_per_octave: ", self.n_scales_per_octave)
        for octave in range(self.n_octaves):
            for scale in range(1, self.n_scales_per_octave):
                # print("i range: ", dog_images[octave][scale].shape[0])
                # print("j range: ", dog_images[octave][scale].shape[1])
                # print("dog shape: ", dog_images[octave][scale].shape)
                for i in range(dog_images[octave][scale].shape[0]):
                    for j in range(dog_images[octave][scale].shape[1]):
                        if self.is_keypoint(
                            dog_images,
                            octave,
                            scale,
                            i,
                            j,
                            threshold,
                            max_tolerance,
                        ):
                            bound_x, bound_y = dog_images[octave][scale].shape
                            new_i, new_j = i, j
                            if not (
                                i - 1 < 0
                                or i + 1 >= bound_x
                                or j - 1 < 0
                                or j + 1 >= bound_y
                            ):
                                hessian_2d, hessian_3d, grad = self.calculate_hessian(
                                    dog_images,
                                    octave,
                                    scale,
                                    i,
                                    j,
                                )

                                d = -1 * np.dot(np.linalg.inv(hessian_3d), grad)

                                new_i = new_i + min(1, max(-1, np.round(d[0])))
                                new_j = new_j + min(1, max(-1, np.round(d[1])))

                            keypoint = KeyPoint()
                            keypoint.pt = (
                                new_j * (2**octave),
                                new_i * (2**octave),
                            )
                            keypoint.octave = octave + scale * (2**8)
                            keypoint.size = (
                                self.initial_sigma
                                * (2 ** (scale / float32(self.n_scales_per_octave)))
                                * (2 ** (octave + 1))
                            )
                            keypoints_cv.append(keypoint)
                            keypoints.append((octave, scale, i, j))
        return keypoints_cv, keypoints

    def calculate_hessian(
        self,
        dog_images: ndarray,
        octave: int,
        scale: int,
        i: int,
        j: int,
    ) -> ndarray:
        """Calculate the hessian matrix for a given pixel.

        Args:
            dog_images (ndarray): Stack of difference of gaussian images
            for each octave and scale.
            octave (int): Octave number.
            scale (int): Scale number.
            i (int): Row number.
            j (int): Column number.
        """
        # for dx in range(-1, 2):
        # for dy in range(-1, 2):
        # print(dog_images[octave][scale][i + dx][j + dy], end=" ")
        # print()

        d_xx = (
            dog_images[octave][scale][i - 1][j]
            - 2 * dog_images[octave][scale][i][j]
            + dog_images[octave][scale][i - 1][j]
        )
        d_yy = (
            dog_images[octave][scale][i][j - 1]
            - 2 * dog_images[octave][scale][i][j]
            + dog_images[octave][scale][i][j + 1]
        )
        d_xy = (1 / 4) * (
            dog_images[octave][scale][i + 1][j + 1]
            - dog_images[octave][scale][i - 1][j + 1]
            - dog_images[octave][scale][i + 1][j - 1]
            + dog_images[octave][scale][i - 1][j - 1]
        )
        d_ss = (
            dog_images[octave][scale - 1][i][j]
            - 2 * dog_images[octave][scale][i][j]
            + dog_images[octave][scale + 1][i][j]
        )
        d_xs = (1 / 4) * (
            dog_images[octave][scale + 1][i + 1][j]
            - dog_images[octave][scale + 1][i - 1][j]
            - dog_images[octave][scale - 1][i + 1][j]
            + dog_images[octave][scale - 1][i - 1][j]
        )
        d_ys = (1 / 4) * (
            dog_images[octave][scale + 1][i][j + 1]
            - dog_images[octave][scale + 1][i][j - 1]
            - dog_images[octave][scale - 1][i][j + 1]
            + dog_images[octave][scale - 1][i][j - 1]
        )
        hessian_2d = np.array([[d_xx, d_xy], [d_xy, d_yy]])
        hessian_3d = np.array(
            [[d_xx, d_xy, d_xs], [d_xy, d_yy, d_ys], [d_xs, d_ys, d_ss]],
        )

        grad = 0.5 * np.array(
            [
                dog_images[octave][scale][i + 1][j]
                - dog_images[octave][scale][i - 1][j],
                dog_images[octave][scale][i][j + 1]
                - dog_images[octave][scale][i][j - 1],
                dog_images[octave][scale + 1][i][j]
                - dog_images[octave][scale - 1][i][j],
            ],
        )

        return hessian_2d, hessian_3d, grad

    def is_local_maximum(
        self,
        dog_images: ndarray,
        octave: int,
        scale: int,
        i: int,
        j: int,
        threshold: int = 5,
    ):
        """Verify if a given pixel is a maxima -> Let's filter it

        Args:
            dog_images (ndarray): Difference of gaussian images for octave and scale
            octave (int): Identification of the octave in the pyramid of gaussian
            scale (int): One of the scales of the octave
            i (int): Pixel x position in the image
            j (int): Pixel y position in the image
        """
        bound_x, bound_y = dog_images[octave][scale].shape
        if i - 1 < 0 or i + 1 >= bound_x or j - 1 < 0 or j + 1 >= bound_y:
            return False, None, None, None

        hessian_2d, hessian_3d, grad = self.calculate_hessian(
            dog_images,
            octave,
            scale,
            i,
            j,
        )

        det_hessian = np.linalg.det(hessian_2d)

        if det_hessian <= 0:
            return False, None, None, None

        trace_hessian = np.trace(hessian_2d)

        a = (trace_hessian**2) / det_hessian
        a_max = (threshold + 1) ** 2 / threshold
        d = -1 * np.dot(np.linalg.inv(hessian_3d), grad)
        response = dog_images[octave][scale][i][j] - 0.5 * np.dot(grad, d)
        # print("Hessian: ", hessian)
        # print("det_hessian: ", det_hessian)
        # print("trace hessian: ", trace_hessian)
        # print("a: ", a)
        # print("a_max: ", a_max)

        return a < a_max, i, j, response

    def filter_keypoints(
        self,
        dog_images: ndarray,
        keypoints: ndarray,
        threshold: float = 5.0,
    ) -> ndarray:
        """Filter keypoints using the hessian matrix.

        Args:
            dog_images (ndarray): Stack of difference of gaussian images
                                                  for each octave and scale.
            keypoints (ndarray): Keypoints found in the difference of gaussian images.
        """
        filtered_keypoints = []
        features = []
        for octave, scale, i, j in keypoints:
            is_local_max, new_i, new_j, response = self.is_local_maximum(
                dog_images,
                octave,
                scale,
                i,
                j,
                threshold,
            )
            if is_local_max:
                orientation = self.calculate_orientation(
                    dog_images,
                    octave,
                    scale,
                    new_i,
                    new_j,
                )
                keypoint = KeyPoint()
                keypoint.response = response
                keypoint.pt = (
                    new_j * (2**octave),
                    new_i * (2**octave),
                )
                keypoint.octave = octave + scale * (2**8)
                keypoint.angle = orientation
                keypoint.size = (
                    self.initial_sigma
                    * (2 ** (scale / float32(self.n_scales_per_octave)))
                    * (2 ** (octave + 1))
                )
                feature_vector = self.make_sift_descriptor_from_keypoint(
                    dog_images,
                    new_i,
                    new_i,
                    octave,
                    scale,
                    orientation,
                )
                features.append(feature_vector)
                filtered_keypoints.append(keypoint)
        self.filtered_keypoints = filtered_keypoints
        self.features = features
        return filtered_keypoints, features

    def calculate_gradient(self, dog_image: ndarray, i: int, j: int) -> ndarray:
        """Calculate the gradient of a keypoint.

        Returns: Polar coordinates of the gradient.

        Args:
            image (ndarray): Image.
            i (int): Row of the keypoint.
            j (int): Column of the keypoint.
        """
        dx = 0.5 * (dog_image[i][j + 1] - dog_image[i][j - 1])
        dy = 0.5 * (dog_image[i + 1][j] - dog_image[i - 1][j])

        e = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        return e, phi

    def convert_keypoints(self, keypoints):
        """Convert keypoint point, size, and octave to input image size"""
        new_keypoints = []
        for kp in keypoints:
            new_kp = KeyPoint()
            new_kp.pt = tuple(0.5 * np.array(kp.pt))
            new_kp.size = 0.5 * kp.size
            new_kp.angle = kp.angle
            new_kp.response = kp.response
            new_kp.octave = (kp.octave & ~255) | ((kp.octave - 1) & 255)
            new_keypoints.append(new_kp)
        return new_keypoints

    def calculate_dominant_histogram(
        self,
        dog_images: ndarray,
        octave: int,
        scale: int,
        x: int,
        y: int,
    ) -> ndarray:
        """Calculate the dominant histogram of a keypoint.
        Returns:
            ndarray: the smothed histogram of the keypoint.
        """
        histogram = np.zeros(36)
        window_scale = 1.5 * self.gaussian_scales[scale]
        window_size = int(np.ceil(2.5 * window_scale))
        n_bins = 36

        e, phi = self.calculate_gradient(dog_images[octave][scale], x, y)

        u = np.arange(-window_size, window_size)
        v = np.arange(-window_size, window_size)
        uu, vv = np.meshgrid(u, v)
        for i in range(2 * window_size):
            for j in range(2 * window_size):
                kappa_phi = n_bins / (2 * np.pi) * phi

                k_0 = int(np.floor(kappa_phi)) % n_bins
                k_1 = int(np.floor(kappa_phi) + 1) % n_bins

                alpha = kappa_phi - np.floor(kappa_phi)
                w = np.exp(
                    -0.5
                    * ((uu[i][j] - x) ** 2 + (vv[i][j] - y) ** 2)
                    / window_scale**2,
                )
                z = w * e
                histogram[k_0] += (1 - alpha) * z
                histogram[k_1] += alpha * z
        histogram = self._smooth_histogram(histogram)
        return histogram

    def _smooth_histogram(self, hist: ndarray) -> ndarray:
        """Smooth the histogram by convolving it with a gaussian kernel"""
        kernel = getGaussianKernel(5, 1)
        return filter2D(hist, -1, kernel)

    def calculate_orientation(
        self,
        dog_images: ndarray,
        octave: int,
        scale: int,
        x: int,
        y: int,
    ) -> ndarray:
        """Calculate the orientation of a keypoint.

        Args:
            image (ndarray): Image.
            i (int): Row of the keypoint.
            j (int): Column of the keypoint.
        """
        histogram = self.calculate_dominant_histogram(dog_images, octave, scale, x, y)
        max_val = np.max(histogram)
        if max_val <= 1e-8:
            return 0

        idx = 10 * np.arange(len(histogram))
        idx = idx.reshape(1, -1)
        histogram = histogram.reshape(idx.shape).flatten()
        idx = idx.flatten()
        # print(histogram)
        # peaks = find_peaks(histogram)
        peaks = interpolate(idx, histogram)
        # print("peaks", peaks)
        return peaks[0]

    def rotate_point(self, u: float, v: float, angle: float) -> tuple[float, float]:
        """Rotate a point by an angle.

        Args:
            u (float): x coordinate of the point.
            v (float): y coordinate of the point.
            angle (float): Angle to rotate by.

        Returns:
            Tuple[float, float]: Rotated point.
        """
        return (
            u * np.cos(angle) - v * np.sin(angle),
            u * np.sin(angle) + v * np.cos(angle),
        )

    def make_sift_descriptor_from_keypoint(
        self,
        dog_images: ndarray,
        x: int,
        y: int,
        octave: int,
        scale: int,
        orientation: float,
        size_factor: float = 5,
        n_spacial_bins: int = 4,
        n_orientation_bins: int = 8,
        f_max: float = 0.2,
        f_scale: float = 512,
    ) -> ndarray:
        """Make a sift descriptor from a keypoint."""

        m, n = dog_images[octave][scale].shape

        window_size = size_factor * scale
        filter_scale = 0.25 * window_size
        cutoff_radius = 2.5 * filter_scale

        u_min = max(int(np.floor(x - cutoff_radius)), 1)
        u_max = min(int(np.ceil(x + cutoff_radius)), m - 2) + 1
        v_min = max(int(np.floor(y - cutoff_radius)), 1)
        v_max = min(int(np.ceil(y + cutoff_radius)), n - 2) + 1

        histogram_3d = np.zeros((n_orientation_bins, n_spacial_bins, n_spacial_bins))

        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                dist = np.sqrt((u - x) ** 2 + (v - y) ** 2)
                # Skip if the point is outside the cutoff radius
                if dist > cutoff_radius:
                    continue

                # Rotate the point (u, v) by orientation
                new_u, new_v = self.rotate_point(
                    u - x,
                    v - y,
                    orientation,
                )
                new_u = (1 / window_size) * new_u
                new_v = (1 / window_size) * new_v

                # Calculate the gradient magnitude and orientation
                e, phi = self.calculate_gradient(dog_images[octave][scale], u, v)
                new_phi = phi - orientation
                while new_phi < 0:
                    new_phi += 2 * np.pi

                w = np.exp(dist**2 / (-2 * filter_scale**2))
                z = w * e

                # Update the histogram
                self.update_histogram(
                    histogram_3d,
                    new_u,
                    new_v,
                    new_phi,
                    z,
                    n_spacial_bins,
                    n_orientation_bins,
                )

        feature_vector = histogram_3d.flatten()

        # Normalize the feature vector
        feature_vector = feature_vector / np.linalg.norm(feature_vector)

        # Clip the feature vector
        feature_vector = np.clip(feature_vector, 0, f_max)

        # Normalize the feature vector again
        feature_vector = feature_vector / np.linalg.norm(feature_vector)

        # Convert to byte
        feature_vector = (f_scale * feature_vector).astype(np.uint8)

        return feature_vector

    def update_histogram(
        self,
        histogram_3d: ndarray,
        u: float,
        v: float,
        phi: float,
        z: float,
        n_spacial_bins: int,
        n_orientation_bins: int,
    ):
        """Update the histogram.

        Args:
            histogram_3d (ndarray): gradient histogram of size (n_orientation_bins,
            n_spacial_bins, n_spacial_bins)
            u (float): normalized x coordinate of the point.
            v (float): normalized y coordinate of the point.
            phi (float): normalized orientation of the point.
            z (float): quantitity to add to the histogram.
            n_spacial_bins (int): number of spacial bins.
            n_orientation_bins (int): number of orientation bins.
        """
        # Calculate the new indices
        new_i = n_spacial_bins * u + 0.5 * (n_spacial_bins - 1)
        new_j = n_spacial_bins * v + 0.5 * (n_spacial_bins - 1)
        new_k = n_orientation_bins * phi / (2 * np.pi)

        # Calculate the indices of the 8 surrounding bins
        i_0 = int(np.floor(new_i))
        i_1 = i_0 + 1
        i = [i_0, i_1]

        j_0 = int(np.floor(new_j))
        j_1 = j_0 + 1
        j = [j_0, j_1]

        k_0 = int(np.floor(new_k)) % n_orientation_bins
        k_1 = (k_0 + 1) % n_orientation_bins
        k = [k_0, k_1]

        # Calculate the weights
        alpha_0 = i_1 - new_i
        alpha_1 = 1 - alpha_0
        alpha = [alpha_0, alpha_1]

        beta_0 = j_1 - new_j
        beta_1 = 1 - beta_0
        beta = [beta_0, beta_1]

        gamma_0 = 1 - (new_k - np.floor(new_k))
        gamma_1 = 1 - gamma_0
        gamma = [gamma_0, gamma_1]

        # Update the histogram
        for ii in i:
            for jj in j:
                for kk in k:
                    for aa in alpha:
                        for bb in beta:
                            for gg in gamma:
                                if (
                                    ii >= 0
                                    and ii < n_spacial_bins
                                    and jj >= 0
                                    and jj < n_spacial_bins
                                ):
                                    histogram_3d[kk][ii][jj] += aa * bb * gg * z
