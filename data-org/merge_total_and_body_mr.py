#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import glob

def load_label_map(filepath):
    """Load a label map from .npy or .nii/.nii.gz file."""
    print(f"Loading label map: {filepath}")
    if filepath.endswith('.npy'):
        data = np.load(filepath)
        affine = None
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.int32)
        affine = img.affine
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    return data, affine

def save_label_map(data, affine, output_path):
    """Save the label map to .npy or .nii.gz format."""
    print(f"Saving merged label map to: {output_path}")
    if output_path.endswith('.npy'):
        np.save(output_path, data)
    elif output_path.endswith('.nii') or output_path.endswith('.nii.gz'):
        img = nib.Nifti1Image(data.astype(np.int32), affine)
        nib.save(img, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")

def remap_body_mr_labels(data):
    """Remap body_mr labels: 1→101, 2→102."""
    print("Remapping body_mr labels: 1→101, 2→102")
    data_remapped = data.copy()
    data_remapped[data == 1] = 101
    data_remapped[data == 2] = 102
    return data_remapped

def merge_label_maps(ref_data, src_data):
    """Merge src_data into ref_data where ref_data is zero."""
    print("Merging label maps...")
    mask = (src_data != 0) & (ref_data == 0)
    merged = ref_data.copy()
    merged[mask] = src_data[mask]
    print(f"Number of voxels added: {np.sum(mask)}")
    return merged

def make_all_filename(original_filename):
    """Replace _oseg or _body with _all in the filename."""
    base = os.path.basename(original_filename)
    if "_oseg" in base:
        return base.replace("_oseg", "_all")
    elif "_body" in base:
        return base.replace("_body", "_all")
    else:
        raise ValueError(f"Filename does not contain _oseg or _body: {original_filename}")

def process_pair(total_path, body_path, output_dir):
    """Process one subject pair of total_mr and body_mr files."""
    total_data, affine = load_label_map(total_path)
    body_data, _ = load_label_map(body_path)
    body_data_remapped = remap_body_mr_labels(body_data)
    merged_data = merge_label_maps(total_data, body_data_remapped)

    # Build output filename based on total_path
    out_filename = make_all_filename(total_path)
    output_path = os.path.join(output_dir, out_filename)
    save_label_map(merged_data, affine, output_path)

def main():
    parser = argparse.ArgumentParser(description="Merge total_mr and body_mr label maps into one combined label map.")
    parser.add_argument('--input_dir', required=True, help='Directory containing subject subdirectories or one directory with all label maps')
    parser.add_argument('--onedir', action='store_true', help='Treat input_dir as a single directory containing all label maps (no subdirectories)')
    args = parser.parse_args()

    if args.onedir:
        # Single directory mode
        print("Running in --onedir mode")
        oseg_files = sorted(glob.glob(os.path.join(args.input_dir, "*_oseg*.nii*")))
        for oseg_path in oseg_files:
            base_no_oseg = os.path.basename(oseg_path).replace("_oseg", "")
            body_candidates = glob.glob(os.path.join(args.input_dir, f"{base_no_oseg.replace('.nii', '').replace('.gz', '')}_body*.nii*"))
            if not body_candidates:
                print(f"WARNING: No body_mr file found for {oseg_path}, skipping.")
                continue
            body_path = body_candidates[0]
            process_pair(oseg_path, body_path, args.input_dir)
    else:
        # Subdirectory-per-subject mode
        print("Running in multi-subdirectory mode")
        subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        for sub_id in subdirs:
            subdir_path = os.path.join(args.input_dir, sub_id)
            oseg_candidates = glob.glob(os.path.join(subdir_path, "*_oseg*.nii*"))
            body_candidates = glob.glob(os.path.join(subdir_path, "*_body*.nii*"))
            if not oseg_candidates or not body_candidates:
                print(f"WARNING: Missing files for {sub_id}, skipping.")
                continue
            oseg_path = oseg_candidates[0]
            body_path = body_candidates[0]
            process_pair(oseg_path, body_path, subdir_path)

    print("All merging complete.")

if __name__ == "__main__":
    main()
