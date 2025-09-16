# Project Title

Lymphoma lesion identification with PET-MR images and Deep Learning (LIPID). 

## Features




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
conda activate gouhfi
```

### Step 2: Install PyTorch >2.0.0 and <2.6.0

- Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the stable PyTorch version based on your OS (Linux, Mac or Windows), package manager (choose `pip` if unsure), language (Python) and compute platform (CUDA 12.8 was tested for this project, but your system requirements might be different).
- For Ubuntu OS with the latest stable PyTorch version through `pip` with at least CUDA 12.8, run this command:
```
pip3 install torch torchvision
```


### Step 3: Install TotalSegmentator

Install the `TotalSegmentator` through `pip` as described [here](https://github.com/wasserth/TotalSegmentator/tree/master?tab=readme-ov-file#installation). If you are too lazy (and on Ubuntu), do this:

```
pip install TotalSegmentator
```

### Step 4: Clone & install the repository locally

```bash
cd path/where/you/want/gouhfi/to/be/installed
git clone https://github.com/mafortin/GOUHFI.git
cd GOUHFI
pip install -e .
```

- The `pip install -e .` command allows you to install the GOUHFI repository in "editable" mode where you can modify the different scripts to your liking.
- **Note**: If you do not have `git` installed on your system, you can manually copy it.

### Step 4: Download the trained model weights

1) A Zenodo link to the trained model weights is included in the repository in the `trained_model/gouhfi-trained-model-weights.md` file or simply with this [link](https://zenodo.org/records/15255556). This might require you to have a Zenodo account (free).
2) Move this `GOUHFI.zip` in the `trained-model` folder before unzipping it.

### Step 5: Unzip the `GOUHFI.zip`

- To unzip `GOUHFI.zip`, use the following command:
```bash
cd trained_model/
unzip GOUHFI.zip
```

- Once unzipped, you should have a folder called `Dataset014_gouhfi` with all trained folds and related files in the `trained_model` folder.

### Step 6: S


```bash
source ~/.bashrc
echo $GOUHFI_HOME
```
- where `~/.bashrc` can be `~/.zshrc`.


### Step 7: Test the installation
    ```

## Usage

See the two documentation files in `/doc/` for the lesion seg and staging.


## License

This project is licensed under the [Apache-2.0 License](http://www.apache.org/licenses/).

## Contact

For questions or feedback, please contact [marc.a.fortin@ntnu.no].
