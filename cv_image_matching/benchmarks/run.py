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

    folders = [
        filename
        for filename in os.listdir(SRC)
        if os.path.isdir(os.path.join(SRC, filename))
    ]
    output_path = "results.csv"
    src = SRC
    errors = {}
    with open(output_path, "w") as f:
        f.write(
            "folder,mean_err_f_own,std_err_f_own,mean_err_f_opencv,std_err_f_opencv,"
            "mean_err_q_own,std_err_q_own,mean_err_t_own,std_err_t_own,"
            "mean_err_q_opencv,std_err_q_opencv,mean_err_t_opencv,std_err_t_opencv\n",
        )
    mean_errors = {}
    std_errors = {}
    for folder in folders:
        errors[folder] = {
            error: []
            for error in [
                "err_f_own",
                "err_f_opencv",
                "err_q_own",
                "err_t_own",
                "err_q_opencv",
                "err_t_opencv",
            ]
        }
        for _ in range(10):
            (
                err_f_own,
                err_f_opencv,
                err_q_own,
                err_t_own,
                err_q_opencv,
                err_t_opencv,
            ) = run_evaluation(
                folder,
                src,
                params_sift,
            )
            errors[folder]["err_q_own"].append(err_q_own)
            errors[folder]["err_q_opencv"].append(err_q_opencv)
            errors[folder]["err_t_own"].append(err_t_own)
            errors[folder]["err_t_opencv"].append(err_t_opencv)
            errors[folder]["err_f_own"].append(err_f_own)
            errors[folder]["err_f_opencv"].append(err_f_opencv)

        mean_errors[folder] = {
            error: np.mean(errors[folder][error])
            for error in [
                "err_f_own",
                "err_f_opencv",
                "err_q_own",
                "err_t_own",
                "err_q_opencv",
                "err_t_opencv",
            ]
        }
        std_errors[folder] = {
            error: np.std(errors[folder][error])
            for error in [
                "err_f_own",
                "err_f_opencv",
                "err_q_own",
                "err_t_own",
                "err_q_opencv",
                "err_t_opencv",
            ]
        }

        with open(output_path, "a") as f:
            f.write(
                f"{folder},{mean_errors[folder]['err_f_own']},{std_errors[folder]['err_f_own']},"
                f"{mean_errors[folder]['err_f_opencv']},{std_errors[folder]['err_f_opencv']},"
                f"{mean_errors[folder]['err_q_own']},{std_errors[folder]['err_q_own']},"
                f"{mean_errors[folder]['err_t_own']},{std_errors[folder]['err_t_own']},"
                f"{mean_errors[folder]['err_q_opencv']},{std_errors[folder]['err_q_opencv']},"
                f"{mean_errors[folder]['err_t_opencv']},{std_errors[folder]['err_t_opencv']}\n",
            )


if __name__ == "__main__":
    main()
