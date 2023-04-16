import os

import numpy as np

from cv_image_matching.benchmarks.fundamental_matrix import run_evaluation

SRC = "data/train"


def main():
    params_sift = {
        "kp_find_threshold": 1,
        "kp_max_tolerance": 0,
        "local_max_threshold": 10,
        "initial_sigma": 1.6,
        "n_scales_per_octave": 3,
        "n_octaves": 8,
        "assumed_blur": 0.5,
        "gaussian_window_histogram": 1.5,
        "num_bins_histogram": 180,
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
    experiments = ["own_sift", "opencv_sift", "superglue", "loftr"]
    error_types = ["err_f", "err_q", "err_t"]

    folders = [
        filename
        for filename in os.listdir(SRC)
        if os.path.isdir(os.path.join(SRC, filename))
    ]
    output_path = "results.csv"
    src = SRC
    errors = {}
    mean_errors = {}
    std_errors = {}
    with open(output_path, "w") as f:
        f.write("scene,")
        for exp in experiments:
            errors[exp] = {}
            mean_errors[exp] = {}
            std_errors[exp] = {}
            for error_type in error_types:
                errors[exp][error_type] = []
                f.write(f"{exp}_mean_{error_type},")
                f.write(f"{exp}_std_{error_type},")
        f.write("\n")

    for folder in folders:
        for _ in range(10):
            error = run_evaluation(folder, src, params_sift, experiments)
            for exp in experiments:
                for error_type in error_types:
                    errors[exp][error_type].append(error[exp][error_type])

        for exp in experiments:
            for error_type in error_types:
                mean_errors[exp][error_type] = np.mean(errors[exp][error_type])
                std_errors[exp][error_type] = np.std(errors[exp][error_type])

        with open(output_path, "a") as f:
            f.write(f"{folder},")
            for exp in experiments:
                for error_type in error_types:
                    f.write(f"{mean_errors[exp][error_type]:.4f},")
                    f.write(f"{std_errors[exp][error_type]:.4f},")
            f.write("\n")


if __name__ == "__main__":
    main()
