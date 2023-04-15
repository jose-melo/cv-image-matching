import argparse

from cv_image_matching.benchmarks.fundamental_matrix import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(r"--show", action="store_true")
    parser.add_argument(r"--idx", type=int, default=-1)
    parser.add_argument(r"--folder", type=str, default="notre_dame_front_facade")
    parser.add_argument(r"--src", type=str, default="data/train")
    parser.add_argument(r"--image_size", type=int, nargs=2, default=(200, 200))

    args, _ = parser.parse_known_args()

    folder = args.folder
    src = args.src
    image_size = args.image_size
    idx = args.idx
    show = args.show

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
        idx,
        image_size,
        show=show,
    )

    print("Own: ", err_q_own, err_t_own, err_f_own)
    print("OpenCV: ", err_q_opencv, err_t_opencv, err_f_opencv)
