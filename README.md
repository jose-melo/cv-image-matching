# Reviving Classical Methods in Computer Vision


![GitHub](https://img.shields.io/github/license/jose-melo/cv-image-matching) ![Lines of code](https://img.shields.io/tokei/lines/github/jose-melo/cv-image-matching)
### A Comprehensive Study on Image Matching
The paper can be found here: [Reviving Classical Methods in Computer Vision](https://github.com/jose-melo/cv-image-matching/files/11243286/Report___Reviving_Classical_Methods_in_Computer_Vision.pdf)


The image matching problem is a significant challenge in computer vision with various applications. Although traditional methods such as local features have been widely used, end-to-end solutions based on deep learning are emerging, presenting new challenges in comparing classical and deep learning-based approaches. This project aims to contribute to the ongoing efforts to advance the field of image matching by studying the parameters of feature detection algorithms and their impact on the accuracy of camera pose estimation. To achieve this, we implemented a SIFT algorithm from scratch and evaluated it against state-of-the-art deep learning-based methods, SuperGlue and LoFTR, using the PhotoTourism dataset. Our results show that the SIFT algorithm performs relatively well in challenging scenarios, even if it is worse than the state-of-the-art methods.

## Folder structure

This project has four main directories:
- ```cv_image_matching```: This directory contains the main project code. It has subdirectories for ```benchmarks```, ```feature_extraction```, ```utils```, and an __init__.py file to mark it as a Python package. The ```run.py``` file is also located in this directory.
     - ```cv_image_matching/benchmarks```: It has the python code to evaluate the experiments (Our SIFT, OpenCV SIFT, SuperGlue and LoFTR)
     - ```cv_image_matching/feature_extraction```: Implementation of feature extraction techniques. The SIFT implementation is in the ```sift.py``` file.
- ```data```: This directory contains the training data for the project. It has a subdirectory named train that contains image data for the matching algorithms.
- ```notebooks```: This directory contains Jupyter notebooks used for development and experimentation. It has subdirectories for profiling and sift, and specific notebooks for keypoints and matching.
    - ```notebooks/keypoints.ipynb```: Notebook with the comparison of key points detection
    - ```notebooks/matching.ipynb```: Notebook with the comparison of the image matching
    - ```notebooks/sift/sift.ipynb```: Notebook for the SIFT implementation
    - ```notebooks/sift/why_match_the_scale.ipynb```: Notebook for to view the Laplacian of Gaussian, and Difference of Gaussians.
- ```results``` : This directory contains the output of the project's experimentation. It has two CSV files named results.csv and results_full.csv, both containing the results of image matching experiments.

Other files in the folder structure include a LICENSE file, a README.md file, and a setup.py file used for packaging and distributing the project.



```
├── cv_image_matching
│   ├── __init__.py
│   ├── benchmarks
│   │   ├──  __init__.py
│   │   ├── fundamental_matrix.py
│   │   ├── run.py
│   │   └── run_script.py
│   ├── feature_extraction
│   │   ├── __init__.py
│   │   ├── peaks.py
│   │   ├── sift.py
│   │   └── test.py
│   ├── run.py
│   └── utils
├── data
│   ├── train
│        ├── ... 
├── notebooks
│   ├── keypoints.ipynb
│   ├── matching.ipynb
│   └── sift
│       ├── sift.ipynb
│       └── why_match_the_scale.ipynb
├── results
│   ├── results.csv
│   └── results_full.csv
├── LICENSE
├── README.md
└── setup.py

```

## SIFT implementation

One of the goals of this project is to implement the SIFT (Scale-Invariant Feature Transform) algorithm from scratch in Python. By building the algorithm from the ground up, we aim to gain a deeper understanding of the full pipeline of classical feature detection and matching methods. This approach allows us to explore and analyze the algorithmic choices and design decisions that affect the performance of the method. Ultimately, the aim is to improve our understanding of the underlying principles of feature detection and matching, and to provide insights into how we can optimize and improve these techniques.  Some results are presented below.

![window_notre_dame_kps](https://user-images.githubusercontent.com/24592687/232333575-459fcec3-4626-474a-8126-e7247d52bef7.png)
![brain_keypoints](https://user-images.githubusercontent.com/24592687/232333623-932d17df-be0d-4836-b2ab-0a910914a8e8.png)

## Benchmarking with other methods

In this project, we aim to estimate the fundamental matrix between two images using the widely used Phototourism dataset as a benchmark for image matching. We will compare classical methods with state-of-the-art algorithms to evaluate their performance in image matching. By doing so, we hope to gain insights into the strengths and limitations of each approach, and to determine which methods are most effective for different types of images and scenarios. Ultimately, the goal is to advance the field of image matching by identifying the most effective techniques and improving our understanding of their underlying principles. Some results are presented below.

![app_lincoln_memorial](https://user-images.githubusercontent.com/24592687/232333641-0f0cf23b-0999-4861-b35f-9f44c715c1ae.png)
![app_notre_dame](https://user-images.githubusercontent.com/24592687/232333644-ed36891b-ad29-4438-9684-20507fccbf3b.png)
![app_sacre_coeur](https://user-images.githubusercontent.com/24592687/232333648-eddf31d6-4531-47d5-9484-6ee6cc92bb9e.png)


## Usage

### Installing the requirements

To install the required libraries:
```
pip install -r requirements.txt
```

### Downloading the data

In order to use the code, it is important to download the data from the <a href="https://www.kaggle.com/competitions/image-matching-challenge-2022/data">Image Challenge<a>

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
