## For retraining a nnUNet model from scratch:

- **Note**: Only relevant if you would decide to retrain from scratch a full 3D U-Net to do the lesion segmentation.

1) Resample the T2w to the PET with the following command:
```
python3 resample_multi_modality_images.py --input_dir /home/marcantf/180n/data/all-exams-subdirs-qaed --output_dir /home/marcantf/180n/data/all-exams-subdirs-qaed-resampled --ref_modality _LYM.nii.gz --resample_modality LYM_label.nii.gz --process resample [--is_label_map]

```

2) Reorganize and rename the files into a single folder with the following command line:
```
 python3 rename_files_nnunet_convention.py --input_dir /home/marcantf/180n/data/training-data-cleaned --output_dir /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/ --mod LYM.nii.gz,LYM_T2.nii.gz --seg_id LYM_label.nii.gz

```
- Now, yo ushould have the PET images as the `0000` one and the T2w as the `0001`.
- If the file naming convention was respected, the IDs should be preserved.

3) Run the train-test split of the dataset using the following command line:
```
python3 train_test_split_datasets.py -i /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labels -tr /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labelsTr -te /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labelsTs --no-subdirs --save-json --json-path /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/ --dataset-named-json
```
- **Note**: The trick here is to use the **labels** folder since it only includes one file per subject in contrast with the iamges folder which contaisn two contrasts/files per subject.
	- That way, you can then move the corresponding multi-contrast images afterwards if you save the .json files for test and train with the following command.

4) Move accordingly the training and test images using the following command:
```
python3 reorganize_nii_files.py -i /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labels -o /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labelsTr -j /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/180n_lesion-seg_v2_train_subjects.json --sub_id_pattern image_yyy

```
- **Note**: You need to run it twice, once for the `test` and once for the `train` split by changing the `.json` produced from the last set accordingly. 

5) Set all label values in the segmentation to 1 instead of the discrete label values by running the following command:
```
python3 remove_keep_reindex_labels.py --input /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labelsTs-not1 --output /home/marcantf/Data/nnunet/raw/Dataset026_180n_retrain/labelsTs --set-to-one all --num-workers 10

```
- To be done for both the `labelsTs` and `labelsTr`.
	- **Note**: Actually, this should have been done earlier, that way we could have avoided doing it twice now, but not the end of the world!
- At this stage, you should be all set data-wise to prepare your nnUNet training!

6) Then, follow the instructions from step #5 (I think) in the nnUNet note in the `misc` folder to know how to proceed to perform a training within the nnUNet framework.