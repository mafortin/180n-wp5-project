import os
import argparse
import SimpleITK as sitk
import numpy as np
import shutil

def resample_image(ref_img_path, target_img_path, output_path, is_label_map=False):

    """Resamples target image to match the matrix size and spacing of reference image using SimpleITK."""
    print(f"[INFO] Starting resampling process...")

    ref_img = sitk.ReadImage(ref_img_path)
    target_img = sitk.ReadImage(target_img_path)

    print(f"[INFO] Reference image shape: {ref_img.GetSize()}")
    print(f"[INFO] Target image shape before resampling: {target_img.GetSize()}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)

    if is_label_map:
        # Use nearest neighbor interpolation for label maps to preserve integer values
        print(f"[INFO] Using nearest neighbor interpolation for label map.")
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # Use cubic interpolation for continuous images
        print(f"[INFO] Using cubic interpolation for continuous image.")
        resampler.SetInterpolator(sitk.sitkBSpline)

    resampled_target_img = resampler.Execute(target_img)

    print(f"[INFO] Target image shape after resampling: {resampled_target_img.GetSize()}")
    sitk.WriteImage(resampled_target_img, output_path)
    print(f"[INFO] Resampled: {os.path.basename(target_img_path)} -> {os.path.basename(output_path)}")

def find_matching_file(directory, substring):
    """Finds a file in the directory that contains the given substring."""
    for filename in os.listdir(directory):
        if substring in filename and filename.endswith(".nii.gz"):
            return os.path.join(directory, filename)
    return None

def process_directory(input_dir, output_dir, ref_substring, resample_substring, process_method, single_dir=False, is_label_map=False):

    """Processes the input directory, handling both subdirectories and single directory cases."""
    if not single_dir and any(os.path.isdir(os.path.join(input_dir, d)) for d in os.listdir(input_dir)):
        
        # Case: Input directory contains subdirectories
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

            # Copy reference image to output directory
            ref_output_path = os.path.join(output_sub_path, os.path.basename(ref_img_path))
            if os.path.abspath(ref_img_path) != os.path.abspath(ref_output_path):
                shutil.copy(ref_img_path, ref_output_path)
                print(f"[INFO] Copied reference image: {os.path.basename(ref_img_path)} -> {os.path.basename(ref_output_path)}")
            else:
                print(f"[INFO] Skipping copy as source and destination are the same: {os.path.basename(ref_img_path)}")

            # Define output file name for the resampled image
            output_path = os.path.join(output_sub_path, os.path.basename(target_img_path))

            if process_method == "resample":
                resample_image(ref_img_path, target_img_path, output_path, is_label_map)
            elif process_method == "coregister":
                print(f"[WARNING] Coregistration option is not yet implemented for {sub}")
            else:
                print(f"[ERROR] Invalid process method: {process_method}")
   
    else:
        # Case: Input directory contains only files or single_dir flag is set
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        print(f"[INFO] Processing single directory: {input_dir}")
        print(f"[INFO] Selected process: {process_method}")

        # Find reference and target images
        ref_img_path = find_matching_file(input_dir, ref_substring)
        target_img_path = find_matching_file(input_dir, resample_substring)

        if not ref_img_path or not target_img_path:
            print(f"[WARNING] Skipping processing (missing ref or target images)")
            return

        # Copy reference image to output directory
        ref_output_path = os.path.join(output_dir, os.path.basename(ref_img_path))
        if os.path.abspath(ref_img_path) != os.path.abspath(ref_output_path):
            shutil.copy(ref_img_path, ref_output_path)
            print(f"[INFO] Copied reference image: {os.path.basename(ref_img_path)} -> {os.path.basename(ref_output_path)}")
        else:
            print(f"[INFO] Skipping copy as source and destination are the same: {os.path.basename(ref_img_path)}")

        # Define output file name for the resampled image
        output_path = os.path.join(output_dir, os.path.basename(target_img_path))

        if process_method == "resample":
            resample_image(ref_img_path, target_img_path, output_path, is_label_map)
        elif process_method == "coregister":
            print(f"[WARNING] Coregistration option is not yet implemented")
        else:
            print(f"[ERROR] Invalid process method: {process_method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for nnU-Net preprocessing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory containing subject subdirectories or files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save processed images. Defaults to input_dir.")
    parser.add_argument("--ref_modality", type=str, required=True, help="Substring to identify reference image (e.g., '0000' for PET).")
    parser.add_argument("--resample_modality", type=str, required=True, help="Substring to identify modality to be resampled (e.g., '0001' for T2w).")
    parser.add_argument("--process", type=str, choices=["resample", "coregister"], required=True, help="Processing method: 'resample' (default) or 'coregister'.")
    parser.add_argument("--single_dir", action="store_true", help="Flag to indicate that the input directory contains only files (no subdirectories).")
    parser.add_argument("--is_label_map", action="store_true", help="Specify whether the file to resample is a label map. If so, nearest neighbor interpolation will be used instead of cubic spline interpolation.")
    
    args = parser.parse_args()
    output_directory = args.output_dir if args.output_dir else args.input_dir  # Default to input_dir if not provided
    process_directory(args.input_dir, output_directory, args.ref_modality, args.resample_modality, args.process, args.single_dir, args.is_label_map)
