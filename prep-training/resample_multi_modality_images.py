import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_image(ref_img_path, target_img_path, output_path):
    """Resamples target image to match the matrix size of reference image."""
    ref_img = nib.load(ref_img_path)
    target_img = nib.load(target_img_path)

    ref_data = ref_img.get_fdata()
    target_data = target_img.get_fdata()

    # Compute resampling factors
    scale_factors = np.array(ref_data.shape) / np.array(target_data.shape)

    # Perform resampling with cubic interpolation
    resampled_target_data = zoom(target_data, scale_factors, order=3)

    # Save the resampled image
    resampled_target_img = nib.Nifti1Image(resampled_target_data, target_img.affine)
    nib.save(resampled_target_img, output_path)

    print(f"✅ Resampled: {os.path.basename(target_img_path)} → {os.path.basename(output_path)}")


def find_matching_file(directory, substring):
    """Finds a file in the directory that contains the given substring."""
    for filename in os.listdir(directory):
        if substring in filename and filename.endswith(".nii.gz"):
            return os.path.join(directory, filename)
    return None


def process_directory(input_dir, output_dir, ref_substring, resample_substring):
    """Processes all subdirectories and resamples target modality using reference modality."""
    subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for sub in subdirs:
        sub_path = os.path.join(input_dir, sub)
        output_sub_path = os.path.join(output_dir, sub)
        os.makedirs(output_sub_path, exist_ok=True)  # Ensure output directory exists

        # Find reference and target images
        ref_img_path = find_matching_file(sub_path, ref_substring)
        target_img_path = find_matching_file(sub_path, resample_substring)

        if not ref_img_path or not target_img_path:
            print(f"⚠️ Skipping {sub} (missing ref or target images)")
            continue

        # Define output file name
        output_path = os.path.join(output_sub_path, os.path.basename(target_img_path).replace(".nii.gz", ".nii.gz"))

        # Resample the target image
        resample_image(ref_img_path, target_img_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample images to match reference matrix size for nnU-Net preprocessing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory containing subject subdirectories.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save resampled images. Defaults to input_dir.")
    parser.add_argument("--ref_modality", type=str, required=True, help="Substring to identify reference image (e.g., '0000' for PET).")
    parser.add_argument("--resample_modality", type=str, required=True, help="Substring to identify modality to be resampled (e.g., '0001' for T2w).")

    args = parser.parse_args()
    output_directory = args.output_dir if args.output_dir else args.input_dir  # Default to input_dir if not provided
    process_directory(args.input_dir, output_directory, args.ref_modality, args.resample_modality)
