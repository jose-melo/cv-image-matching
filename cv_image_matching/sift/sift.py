from numpy import ndarray
from cv2 import GaussianBlur, resize, getGaussianKernel, filter2D, subtract
import numpy as np


class SIFT(object):
    def __init__(
        self,
        initial_sigma: float = 1.6,
        n_scales_per_octave: int = 3,
        n_octaves: int = 4,
    ) -> None:
        self.initial_sigma = initial_sigma
        self.n_scales_per_octave = n_scales_per_octave
        self.n_octaves = n_octaves

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
        self, dog_images: ndarray, octave: int, scale: int, i: int, j: int, threshold: float = 0.04
    ) -> bool:
        """Check if a pixel is a keypoint (local maximum)."""

        bound_x, bound_y = dog_images[octave][scale].shape
        for x in np.arange(-3, 3):
            for y in np.arange(-3, 3):
                for z in np.arange(-1, 1):
                    if 0 <= i + x < bound_x and 0 <= j + y < bound_y:
                        if (
                            dog_images[octave][scale][i][j]
                            < dog_images[octave][scale + z][i + x][j + y]
                        ):
                            return False
        return True

    def find_keypoints(self, dog_images: ndarray) -> ndarray:
        """Find the keypoints in the difference of gaussian images.

        Args:
            dog_images (ndarray): Stack of difference of gaussian images
            for each octave and scale.
        """
        keypoints = []
        for octave in range(self.n_octaves):
            for scale in range(1, self.n_scales_per_octave):
                for i in range(1, dog_images[octave][scale].shape[0] - 1):
                    for j in range(1, dog_images[octave][scale].shape[1] - 1):
                        if self.is_keypoint(dog_images, octave, scale, i, j):
                            keypoints.append((octave, scale, i, j))
        return keypoints

    def calculate_hessian(
        self, dog_images: ndarray, octave: int, scale: int, i: int, j: int
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
        hessian = np.zeros((3, 3))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    hessian[x + 1][y + 1] += dog_images[octave][scale + z][i + x][j + y]
        return hessian

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

        hessian = self.calculate_hessian(dog_images, octave, scale, i, j)

        det_hessian = np.linalg.det(hessian)

        if det_hessian <= 0:
            return False

        trace_hessian = np.trace(hessian)

        a = (trace_hessian**2) / det_hessian
        a_max = (threshold + 1) ** 2 / threshold

        return a < a_max

    def filter_keypoints(
        self, dog_images: ndarray, keypoints: ndarray, threshold: int = 5
    ) -> ndarray:
        """Filter keypoints using the hessian matrix.

        Args:
            dog_images (ndarray): Stack of difference of gaussian images
                                                  for each octave and scale.
            keypoints (ndarray): Keypoints found in the difference of gaussian images.
        """
        filtered_keypoints = []
        for octave, scale, i, j in keypoints:
            if self.is_local_maximum(dog_images, octave, scale, i, j, threshold):
                filtered_keypoints.append((octave, scale, i, j))
        return filtered_keypoints

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

    def calculate_dominant_histogram(
        self, dog_images: ndarray, octave: int, scale: int, x: int, y: int
    ) -> ndarray:
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
                    / window_scale**2
                )
                z = w * e

                histogram[k_0] += (1 - alpha) * z
                histogram[k_1] += alpha * z
        print(histogram)
        histogram = self._smooth_histogram(histogram)
        print(histogram)
        return histogram

    def _smooth_histogram(self, hist: ndarray) -> ndarray:
        """Smooth the histogram by convolving it with a gaussian kernel"""
        kernel = getGaussianKernel(5, 1)
        return filter2D(hist, -1, kernel)

    def calculate_orientation(
        self, dog_images: ndarray, octave: int, scale: int, x: int, y: int
    ) -> ndarray:
        """Calculate the orientation of a keypoint.

        Args:
            image (ndarray): Image.
            i (int): Row of the keypoint.
            j (int): Column of the keypoint.
        """
        histogram = self.calculate_dominant_histogram(dog_images, octave, scale, x, y)
        return histogram
