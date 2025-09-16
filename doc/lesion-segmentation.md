# Lymphoma lesion segmentation - SOP



1) Files need to be organized in a specific way which is the following:
<pre> /dataset-for-lesion-seg 
├── sub001 
|      ├── 001_LYM.nii.gz (PET AC image)
|      └── 001_LYM_T2.nii.gz (T2w MR image)
├── sub002
|      ├── 002_LYM.nii.gz
|      └── 002_LYM_T2.nii.gz
├── sub003
...
 </pre>

- **Note**: Both the PET and T2w images are required since the U-Net was trained with both contrasts.


2) Your images need to be resampled and coregistered to the same space. This can be done with the following script executed with the command below.
	- **Note**: All these command line examples need to be run inside the `180n` code directory (and inside your virtual python environment).
```
python3 /prep-training/resample_multi_modality_images.py --input_dir XX --output_dir YY --ref_modality LYM.nii.gz --resample_modality LYM_T2.nii.gz --process {resample,coregister} [--single_dir --is_label_map]
```

- where:
	- `--input_dir`/`output_dir` are both the input and output directories respectively.
	- `--ref_modality`/`--resample_modality` are the image that you want to use as the reference/fixed one  and the one to be "moved"/resampled/coregistered to the fixed one respectively.
		- For both, you simply need to input a substring to identify the modality (e.g., here the `LYM_T2.nii.gz` tells it to resample the T2w to the PET (`LYM.nii.gz`)).
		- **Note**: If you need to resample/coregister more than one image, you need to run this script for each image to be processed.
	- `--process` is used to specify whether you want to do `resample` or `coregister` between both images.
	- ``--single_dir`` is a flag to specify whether your input directory contains subfolders or not.
		- Can be a useful flag if you want to process only one subdirectory for instance.
	- `--is_label_map` will use Nearest-Neighbours (NN) instead of 3D linear interpolation for resampling if the ``--resample_modality`` is a label map and not an image.

