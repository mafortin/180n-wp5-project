#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import glob
import scipy.ndimage as ndi

# -------------------------------
# Loading / Saving
# -------------------------------

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

# -------------------------------
# Label manipulation
# -------------------------------

def remap_body_mr_labels(data):
    """Remap body_mr labels: 1 -> 101, 2 -> 102."""
    print("Remapping body_mr labels: 1 -> 101, 2 -> 102")
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

def identify_and_label_head(data, brain_data, extremities_label=102, head_label=103, brain_label=50):
    """
    Identify topmost connected component of extremities and relabel as head.
    Verify that brain_label in brain_data is fully inside this head region.
    """
    extremities_mask = (data == extremities_label)
    labeled, num = ndi.label(extremities_mask)

    if num == 0:
        print("No extremities found for head identification.")
        return data

    # Identify component with highest z-extent
    max_z_per_component = []
    for comp_id in range(1, num + 1):
        coords = np.argwhere(labeled == comp_id)
        max_z = coords[:, 2].max()  # Assuming z-axis is last dim
        max_z_per_component.append((comp_id, max_z))

    head_component = max(max_z_per_component, key=lambda x: x[1])[0]
    head_mask = (labeled == head_component)

    # Brain label mask
    brain_mask = (brain_data == brain_label)

    # Verification: brain completely inside head_mask
    if np.any(brain_mask) and not np.all(head_mask[brain_mask]):
        print("WARNING: Brain is not fully inside the identified head component. Skipping head relabel.")
        return data

    # Apply head label
    data[head_mask] = head_label
    print(f"Head identified as component {head_component}, relabeled to {head_label}.")
    return data




def _trunk_stats(data, trunc_label=101):
    """Return trunk x-center and z-extent."""
    trunk_mask = (data == trunc_label)
    if not np.any(trunk_mask):
        raise ValueError("Trunk (label 101) not found.")
    xs, ys, zs = np.nonzero(trunk_mask)
    return {
        "x_center": xs.mean(),       # sagittal midline anchored to trunk
        "z_min": int(zs.min()),      # inferior end of trunk
        "z_max": int(zs.max()),
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
    }




def identify_and_label_legs(data, trunc_label=101, extremities_label=102,
                            left_leg_label=106, right_leg_label=107,
                            trunc_percentile=5):
    """
    Label legs from extremities (102) that are INFERIOR to a robust lower
    bound of the trunk (percentile of trunk z-distribution).
    """
    trunk_mask = (data == trunc_label)
    if not np.any(trunk_mask):
        raise ValueError("Trunk (label 101) not found.")
    xs, ys, zs = np.nonzero(trunk_mask)
    x_center = xs.mean()
    z_thresh = np.percentile(zs, trunc_percentile)

    ext_mask = (data == extremities_label)
    if not np.any(ext_mask):
        print("No extremities left to classify as legs.")
        return data

    X = np.arange(data.shape[0])[:, None, None]
    Z = np.arange(data.shape[2])[None, None, :]

    legs_mask = ext_mask & (Z < z_thresh)
    if not np.any(legs_mask):
        print("No leg voxels found below trunk threshold.")
        return data

    right_mask = legs_mask & (X <  x_center)
    left_mask  = legs_mask & (X >= x_center)

    data[left_mask]  = left_leg_label
    data[right_mask] = right_leg_label

    print(f"Labeled legs: L={int(left_mask.sum())} vox, R={int(right_mask.sum())} vox (z_thresh={z_thresh}).")
    return data






def identify_and_label_arms(data, trunc_label=101, extremities_label=102,
                            left_arm_label=104, right_arm_label=105):
    """
    Label arms AFTER legs and head.
    Arms are defined strictly as the REMAINING extremities (still 102),
    split left/right by the trunk sagittal midline.
    """
    stats = _trunk_stats(data, trunc_label=trunc_label)

    # only from original extremities that haven't been turned into head/legs
    arms_mask = (data == extremities_label)
    if not np.any(arms_mask):
        print("No extremities left to classify as arms.")
        return data

    X = np.arange(data.shape[0])[:, None, None]
    left_mask  = arms_mask & (X >=  stats["x_center"])
    right_mask = arms_mask & (X < stats["x_center"])

    data[left_mask]  = left_arm_label
    data[right_mask] = right_arm_label

    print(f"Labeled arms: L={int(left_mask.sum())} vox, R={int(right_mask.sum())} vox.")
    return data


# -------------------------------
# Filename utilities
# -------------------------------

def make_all_filename(original_filename):
    """Replace _oseg or _body with _all in the filename."""
    base = os.path.basename(original_filename)
    if "_oseg" in base:
        return base.replace("_oseg", "_all")
    elif "_body" in base:
        return base.replace("_body", "_all")
    else:
        raise ValueError(f"Filename does not contain _oseg or _body: {original_filename}")

# -------------------------------
# Processing
# -------------------------------

def process_pair(total_path, body_path, output_dir):
    """Process one subject pair of total_mr and body_mr files."""
    total_data, affine = load_label_map(total_path)
    body_data, _ = load_label_map(body_path)

    # Remap and identify head (103)
    body_data_remapped = remap_body_mr_labels(body_data)
    body_data_remapped = identify_and_label_head(
        body_data_remapped,
        brain_data=total_data,
        extremities_label=102,
        head_label=103,
        brain_label=50
    )

    # 1) Legs first (from 102 below trunk) → 106/107
    body_data_remapped = identify_and_label_legs(
        body_data_remapped,
        trunc_label=101,
        extremities_label=102,
        left_leg_label=106,
        right_leg_label=107
    )

    # 2) Arms next (STRICTLY remaining 102) → 104/105
    body_data_remapped = identify_and_label_arms(
        body_data_remapped,
        trunc_label=101,
        extremities_label=102,
        left_arm_label=104,
        right_arm_label=105
    )

    # Merge into total map
    merged_data = merge_label_maps(total_data, body_data_remapped)


    # Build output filename based on total_path
    out_filename = make_all_filename(total_path)
    output_path = os.path.join(output_dir, out_filename)
    save_label_map(merged_data, affine, output_path)

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge total_mr and body_mr label maps into one combined label map.")
    parser.add_argument('--input_dir', required=True, help='Directory containing subject subdirectories or one directory with all label maps')
    parser.add_argument('--onedir', action='store_true', help='Treat input_dir as a single directory containing all label maps (no subdirectories)')
    args = parser.parse_args()

    if args.onedir:
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
