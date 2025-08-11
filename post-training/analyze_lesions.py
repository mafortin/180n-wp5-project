#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import csv
from scipy.ndimage import label

def find_files(input_dir, pattern, onedir=False):
    matched_files = []
    if onedir:
        for fname in os.listdir(input_dir):
            if pattern in fname:
                matched_files.append(os.path.join(input_dir, fname))
    else:
        for sub in os.listdir(input_dir):
            sub_path = os.path.join(input_dir, sub)
            if os.path.isdir(sub_path):
                for fname in os.listdir(sub_path):
                    if pattern in fname:
                        matched_files.append(os.path.join(sub_path, fname))
    return matched_files

def get_pet_path(mask_path):
    base_dir = os.path.dirname(mask_path)
    fname_noext = os.path.splitext(os.path.splitext(os.path.basename(mask_path))[0])[0]  # remove .nii.gz
    # Keep everything up to the first occurrence of "_LYM"
    if "_LYM" in fname_noext:
        subj_prefix = fname_noext.split("_LYM")[0]
        return os.path.join(base_dir, f"{subj_prefix}_LYM.nii.gz")
    else:
        # If "_LYM" not found, fallback: just replace extension
        return os.path.join(base_dir, fname_noext + "_LYM.nii.gz")


def process_label_map(filepath, save_instances=False, mask_pattern="_LYM_label.nii.gz"):
    """Process one binary label map: count lesions, compute volumes, and SUV stats."""
    img = nib.load(filepath)
    data = img.get_fdata()
    voxel_volume_mm3 = np.prod(img.header.get_zooms())  # mm³ per voxel

    # Binary mask
    binary_mask = (data > 0).astype(np.uint8)

    # Connected components
    labeled_array, num_features = label(binary_mask)

    # Try loading PET image
    pet_path = get_pet_path(filepath)
    pet_available = os.path.exists(pet_path)
    if pet_available:
        pet_img = nib.load(pet_path)
        pet_data = pet_img.get_fdata()
    else:
        print(f"PET image not found for: {filepath} (expected: {pet_path})")
        pet_data = None

    # Calculate lesion metrics
    volumes = []
    for lesion_id in range(1, num_features + 1):
        lesion_mask = (labeled_array == lesion_id)
        voxel_count = np.sum(lesion_mask)
        volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0  # mm³ → mL

        metrics = {
            "lesion_id": lesion_id,
            "voxel_count": voxel_count,
            "volume_ml": volume_ml,
            "SUV_max": "",
            "SUV_mean": "",
            "SUV_95percentile": ""
        }

        if pet_available:
            lesion_pet_values = pet_data[lesion_mask]
            metrics["SUV_max"] = float(np.max(lesion_pet_values))
            metrics["SUV_mean"] = float(np.mean(lesion_pet_values))
            metrics["SUV_95percentile"] = float(np.percentile(lesion_pet_values, 95))

        volumes.append(metrics)

    # Sort by volume
    volumes.sort(key=lambda x: x["volume_ml"], reverse=True)

    # Save instance map if requested
    if save_instances:
        out_path = filepath.replace(".nii.gz", "_inst.nii.gz")
        out_img = nib.Nifti1Image(labeled_array.astype(np.int32), img.affine, img.header)
        nib.save(out_img, out_path)
        print(f"Saved instance lesion segmentation to: {out_path}")

    return num_features, volumes


def save_csv(volumes, filepath):
    """Save lesion metrics to CSV file."""
    csv_path = filepath.replace(".nii.gz", "_lesions.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "lesion_id", "voxel_count", "volume_ml",
            "SUV_max", "SUV_mean", "SUV_95percentile"
        ])
        writer.writeheader()
        writer.writerows(volumes)
    print(f"Saved lesion metrics to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Count lesions, calculate volumes, and extract SUV metrics from PET images.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing subject subdirs or single dir with files.")
    parser.add_argument("--onedir", action="store_true", help="Treat input_dir as containing all label maps directly.")
    parser.add_argument("--pattern", default="LYM_label.nii.gz", help="Substring to identify the label map to use.")
    parser.add_argument("--save", action="store_true", help="Save instance-labeled NIfTI files with '_inst' suffix.")
    parser.add_argument("--topn", default=10, help="Number of largest lesions to print. Use 'all' for all lesions.")
    args = parser.parse_args()

    files = find_files(args.input_dir, args.pattern, onedir=args.onedir)
    if not files:
        print(f"No files found matching pattern '{args.pattern}' in {args.input_dir}")
        return

    for fpath in files:
        print(f"\nProcessing: {fpath}")
        num_lesions, volumes = process_label_map(fpath, save_instances=args.save, mask_pattern=args.pattern)

        # Save all metrics to CSV
        save_csv(volumes, fpath)

        # Print summary (only top N lesions)
        if str(args.topn).lower() == "all":
            show_volumes = volumes
        else:
            show_volumes = volumes[:int(args.topn)]

        print(f"  Total number of lesions: {num_lesions} (showing {len(show_volumes)} largest)")
        for v in show_volumes:
            print(f"    Lesion {v['lesion_id']}: {v['volume_ml']:.2f} mL/cm3")

if __name__ == "__main__":
    main()
