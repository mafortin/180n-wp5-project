#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label

def find_files(input_dir, pattern, onedir=False):
    """Find label map files matching the pattern."""
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

def process_label_map(filepath, save_instances=False):
    """Process one binary label map: count components and compute volumes."""
    img = nib.load(filepath)
    data = img.get_fdata()
    voxel_volume_mm3 = np.prod(img.header.get_zooms())  # mm³ per voxel

    # Binary mask
    binary_mask = (data > 0).astype(np.uint8)

    # Connected components
    labeled_array, num_features = label(binary_mask)

    # Calculate volumes
    volumes = []
    for comp_id in range(1, num_features + 1):
        voxel_count = np.sum(labeled_array == comp_id)
        volumes.append({
            "component_id": comp_id,
            "voxel_count": voxel_count,
            "volume_mm3": voxel_count * voxel_volume_mm3
        })

    # Sort by volume (largest first)
    volumes.sort(key=lambda x: x["volume_mm3"], reverse=True)

    # Save instance map if requested
    if save_instances:
        out_path = filepath.replace(".nii.gz", "_inst.nii.gz")
        out_img = nib.Nifti1Image(labeled_array.astype(np.int32), img.affine, img.header)
        nib.save(out_img, out_path)
        print(f"Saved instance-labeled map to: {out_path}")

    return num_features, volumes

def main():
    parser = argparse.ArgumentParser(description="Count connected components and calculate volumes in binary lesion masks.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing subject subdirs or single dir with files.")
    parser.add_argument("--onedir", action="store_true", help="Flag: treat input_dir as containing all label maps directly.")
    parser.add_argument("--pattern", default="LYM_label.nii.gz", help="Substring to identify the label map to use.")
    parser.add_argument("--save", action="store_true", help="Save instance-labeled NIfTI files with '_inst' suffix.")
    args = parser.parse_args()

    files = find_files(args.input_dir, args.pattern, onedir=args.onedir)
    if not files:
        print(f"No files found matching pattern '{args.pattern}' in {args.input_dir}")
        return

    for fpath in files:
        print(f"\nProcessing: {fpath}")
        num_features, volumes = process_label_map(fpath, save_instances=args.save)

        print(f"  Number of connected components (lesions): {num_features}")
        for v in volumes:
            print(f"    Component {v['component_id']}: {v['voxel_count']} voxels, {v['volume_mm3']:.2f} mm³")

if __name__ == "__main__":
    main()
