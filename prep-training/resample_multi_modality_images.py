import os
import argparse
import SimpleITK as sitk
import numpy as np

def resample_image(ref_img_path, target_img_path, output_path):
    """Resamples target image to match the matrix size and spacing of reference image using SimpleITK."""
    ref_img = sitk.ReadImage(ref_img_path)
    target_img = sitk.ReadImage(target_img_path)

    print(f"Reference image shape: {ref_img.GetSize()}")
    print(f"Target image shape before resampling: {target_img.GetSize()}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkBSpline)  # Cubic interpolation
    resampled_target_img = resampler.Execute(target_img)

    print(f"Target image shape after resampling: {resampled_target_img.GetSize()}")
    sitk.WriteImage(resampled_target_img, output_path)
    print(f"[INFO] Resampled: {os.path.basename(target_img_path)} -> {os.path.basename(output_path)}")

def find_matching_file(directory, substring):
    """Finds a file in the directory that contains the given substring."""
    for filename in os.listdir(directory):
        if substring in filename and filename.endswith(".nii.gz"):
            return os.path.join(directory, filename)
    return None

def process_directory(input_dir, output_dir, ref_substring, resample_substring, process_method):
    """Processes all subdirectories and applies the specified process (resample or coregister)."""
    subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for sub in subdirs:
        sub_path = os.path.join(input_dir, sub)
        output_sub_path = os.path.join(output_dir, sub)
        os.makedirs(output_sub_path, exist_ok=True)  # Ensure output directory exists

        print(f"[INFO] Processing subject: {sub}")
        print(f"[INFO] Selected process: {process_method}")

        # Find reference and target images
        ref_img_path = find_matching_file(sub_path, ref_substring)
        target_img_path = find_matching_file(sub_path, resample_substring)

        if not ref_img_path or not target_img_path:
            print(f"[WARNING] Skipping {sub} (missing ref or target images)")
            continue

        # Define output file name
        output_path = os.path.join(output_sub_path, os.path.basename(target_img_path))

        if process_method == "resample":
            resample_image(ref_img_path, target_img_path, output_path)
        elif process_method == "coregister":
            print(f"[WARNING] Coregistration option is not yet implemented for {sub}")
        else:
            print(f"[ERROR] Invalid process method: {process_method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for nnU-Net preprocessing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory containing subject subdirectories.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save processed images. Defaults to input_dir.")
    parser.add_argument("--ref_modality", type=str, required=True, help="Substring to identify reference image (e.g., '0000' for PET).")
    parser.add_argument("--resample_modality", type=str, required=True, help="Substring to identify modality to be resampled (e.g., '0001' for T2w).")
    parser.add_argument("--process", type=str, choices=["resample", "coregister"], required=True, help="Processing method: 'resample' (default) or 'coregister'.")

    args = parser.parse_args()
    output_directory = args.output_dir if args.output_dir else args.input_dir  # Default to input_dir if not provided
    process_directory(args.input_dir, output_directory, args.ref_modality, args.resample_modality, args.process)
