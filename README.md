# Reviving Classical Methods in Computer Vision
### A Comprehensive Study on Image Matching

## Usage

### Running a single experiment with customized params

The script can be executed using the following command:
```bash
python cv_image_matching/benchmarks/run_script.py [--show] [--idx INDEX] [--folder FOLDER] [--src SRC] [--image_size WIDTH HEIGHT] [--experiments EXPERIMENTS]
```
where:
```
    --show: (optional) If present, displays the matched images along with the computed fundamental matrix.
    --idx INDEX: (optional) Index of the image pair to evaluate. Default is -1, which means that all image pairs in the specified folder will be evaluated.
    --folder FOLDER: (optional) Name of the folder containing the image pairs to evaluate. Default is "notre_dame_front_facade".
    --src SRC: (optional) Path to the folder containing the source images. Default is "data/train".
    --image_size WIDTH HEIGHT: (optional) Size of the images to be used for evaluation. Default is (200, 200).
    --experiments EXPERIMENTS: (optional) List of experiments to run. Available options are "own_sift", "opencv_sift", and "superglue". Default is ["own_sift", "opencv_sift", "superglue"].    
```

### Running the benchmark experiment

The script can be executed using the following command:
```bash
python cv_image_matching/benchmarks/run.py
```
