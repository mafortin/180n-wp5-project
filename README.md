# 180N-WP5: Lymphoma segmentation

Lymphoma lesion identification and segmentation with PET-MR images and Deep Learning.

Author: Marc-Antoine Fortin


## Installation

### Step 1: Create a Python virtual environment

- As for any Python project, we highly recommend you to install this project inside a virtual environment. Whether you use pip, anaconda or miniconda is up to you, but the steps below use conda. Relevant links related to [conda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) in general or [its installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html) for Ubuntu distributions (OS dependent).

- If you are using conda, you can use the following command: 
```bash
conda create --name 180n-wp5 python=3.10 
```
- `180n-wp5` in the above command line is the name of the virtual environment and can be replaced by anything else if preferred.
- Once your python virtual environment is created, you need to execute the remaining steps inside this virtual environment. Thus, activate the virtual environment by typing:

```bash
conda activate 180n-wp5
```

### Step 2: Install PyTorch (>2.0.0 and <2.6.0)

Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) and choose your OS, Python and CUDA version.

If you have an NVIDIA GPU and want CUDA support (recommended), install PyTorch with CUDA 12.x, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install TotalSegmentator

Install the `TotalSegmentator` through `pip` as described [here](https://github.com/wasserth/TotalSegmentator/tree/master?tab=readme-ov-file#installation). If you are too lazy (and on Ubuntu), do this:

```
pip install TotalSegmentator
```

### Step 4: Clone & install the repository locally

```bash
cd path/where/you/want/the/project/to/be/installed
git clone https://github.com/mafortin/180n-wp5-project.git
cd 180n-wp5-project
```

- **Note**: If you do not have `git` installed on your system, you can manually download the zipped repository with the green code button at the top of the repository.


### Step 5: nnUNet installation

Since the `TotalSegmentator` uses the nnUNet framework, I would believe that you don't need to also install the `nnUNet` framework by itself in order to run new trainings or inference, but I haven't tried myself. Thus, I expect this to current installation to be working, but I haven't thoroughly tested it. 

## Usage

See the two documentation files in `/doc/` for the lesion segmentation and staging scripts.


## License

This project is licensed under the [Apache-2.0 License](http://www.apache.org/licenses/).

## Contact

For questions or feedback, please contact [marc.a.fortin@ntnu.no].
