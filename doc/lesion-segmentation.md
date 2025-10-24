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


3) Now that the image data is ready, you need to make sure that they follow the naming convention as requested by the nnUNet. This cna be performed using the script called `rename_file_nnunet_convention.py` with the following flags:

```
python3 rename_files_nnunet_convention.py --input_dir /home/marcantf/180n/data/training-data-cleaned --output_dir /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/images --mod LYM.nii.gz,LYM_T2.nii.gz --seg_id LYM_label.nii.gz

```

- This will add the `_0000` and `_0001` suffixes for the PET and T2w images respectively will keeping the rest of the name intact. The rest of the filename doesn't matter, what matter is the proper suffixes for each modality since it was trained on the PET **and** T1w images, in that order.


4) Finally, the images should eb ready to be segmented running the following command line:


```
 python3 inf_postpro_eval_nnunet.py --dataset_id Dataset026_180n_retrain --input_dir /home/marcantf/180n/results/test-lesion-segs/imagesTs --config 3d_fullres --gt_dir /home/marcantf/180n/results/test-lesion-segs/gt-test-subs --eval --np 8 --trainer my_nnUNetTrainer --plan plans4lesion-seg
```

- **Note**: This requires the user to (i) have an installed and functioning version of the nnUNet on their computer and (ii), highly probably, also a python virtual environment to run the inference. The `180n` virtual environment *might* actually work since it includes the requirements from the `TotalSegmentator` using also he nnUNet, but I haven't tested it myself. 
- The `--trainer` flag value can be one of the three following ones depending on which trained U-Net you want to use for inference:
	- `my_nnUNetTrainer` [base LR = 3e-4]
	- `my_nnUNetTrainer_baseLR_1e4` [base LR = 1e-4]
	- `my_nnUNetTrainer_baseLR_5e5` [base LR = 5e-5]


