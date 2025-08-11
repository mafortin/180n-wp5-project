#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
import csv
from scipy.ndimage import label

# Organ label mapping
ALL_MR_LABELS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_left", 11: "lung_right", 12: "esophagus", 13: "small_bowel", 14: "duodenum",
    15: "colon", 16: "urinary_bladder", 17: "prostate", 18: "sacrum", 19: "vertebrae",
    20: "intervertebral_discs", 21: "spinal_cord", 22: "heart", 23: "aorta",
    24: "inferior_vena_cava", 25: "portal_vein_and_splenic_vein", 26: "iliac_artery_left",
    27: "iliac_artery_right", 28: "iliac_vena_left", 29: "iliac_vena_right",
    30: "humerus_left", 31: "humerus_right", 32: "scapula_left", 33: "scapula_right",
    34: "clavicula_left", 35: "clavicula_right", 36: "femur_left", 37: "femur_right",
    38: "hip_left", 39: "hip_right", 40: "gluteus_maximus_left", 41: "gluteus_maximus_right",
    42: "gluteus_medius_left", 43: "gluteus_medius_right", 44: "gluteus_minimus_left",
    45: "gluteus_minimus_right", 46: "autochthon_left", 47: "autochthon_right",
    48: "iliopsoas_left", 49: "iliopsoas_right", 50: "brain", 101: "trunc", 102: "extremities"
}

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
    


def analyze_lesions(label_map_path, save_instances=False, mask_pattern="_LYM_label.nii.gz",
                    anat_pattern="_all.nii.gz"):
    # Load lesion mask
    img = nib.load(label_map_path)
    data = img.get_fdata().astype(np.uint8)

    # Instance segmentation
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_array, num_features = label(data, structure=struct)

    # Save instances if requested
    if save_instances:
        inst_path = label_map_path.replace(mask_pattern, mask_pattern.replace(".nii.gz", "_inst.nii.gz"))
        nib.save(nib.Nifti1Image(labeled_array, img.affine, img.header), inst_path)

    # Try loading PET image
    pet_path = get_pet_path(label_map_path)
    if os.path.exists(pet_path):
        pet_img = nib.load(pet_path)
        pet_data = pet_img.get_fdata()
    else:
        print(f"WARNING: PET image not found for {label_map_path}. SUV metrics will be skipped.")
        pet_data = None



    # Load anatomy segmentation if available
    anat_path = None
    for f in os.listdir(os.path.dirname(label_map_path)):
        if "_all.nii.gz" in f:
            anat_path = os.path.join(os.path.dirname(label_map_path), f)
            break

    anat_data = None
    if anat_path and os.path.exists(anat_path):
        anat_img = nib.load(anat_path)
        anat_data = anat_img.get_fdata().astype(int)
    else:
        print(f"WARNING: Anatomy segmentation not found for {label_map_path}")


    voxel_volume_ml = np.prod(img.header.get_zooms()) / 1000.0
    lesions_info = []

    for lesion_id in range(1, num_features + 1):
        lesion_mask = (labeled_array == lesion_id)
        voxel_count = np.sum(lesion_mask)
        volume_ml = voxel_count * voxel_volume_ml

        # PET stats
        if pet_data is not None:
            suv_values = pet_data[lesion_mask]
            suv_max = float(np.max(suv_values))
            suv_mean = float(np.mean(suv_values))
            suv_95p = float(np.percentile(suv_values, 95))
        else:
            suv_max = suv_mean = suv_95p = 0.0


        # Organ overlap
        organ_info = [("None", 0.0), ("None", 0.0), ("None", 0.0)]
        if anat_data is not None:
            lesion_organs_raw, counts_raw = np.unique(anat_data[lesion_mask], return_counts=True)
            mask = lesion_organs_raw != 0
            lesion_organs = lesion_organs_raw[mask]
            counts = counts_raw[mask]


            if len(lesion_organs) == 0:
                print(f"WARNING: Lesion {lesion_id} in {os.path.basename(label_map_path)} has no overlap with any organ label.")
            else:
                total_vox = np.sum(counts)
                percents = (counts / total_vox) * 100
                sort_idx = np.argsort(percents)[::-1]
                lesion_organs = lesion_organs[sort_idx]
                percents = percents[sort_idx]

                top_n = min(3, len(lesion_organs))
                organ_info = []
                for i in range(top_n):
                    organ_name = ALL_MR_LABELS.get(int(lesion_organs[i]), "unknown")
                    organ_info.append((organ_name, round(float(percents[i]), 2)))
                while len(organ_info) < 3:
                    organ_info.append(("None", 0.0))

                if organ_info[0][1] < 50.0:
                    print(f"WARNING: Lesion {lesion_id} in {os.path.basename(label_map_path)} "
                          f"has no single organ covering >50% (top={organ_info[0][0]}, {organ_info[0][1]}%).")

                # Fully inside one organ -> zero out others
                if organ_info[0][1] == 100.0:
                    organ_info[1] = ("None", 0.0)
                    organ_info[2] = ("None", 0.0)

        lesions_info.append({
            "lesion_id": lesion_id,
            "voxel_count": int(voxel_count),
            "volume_ml": volume_ml,
            "SUV_max": suv_max,
            "SUV_mean": suv_mean,
            "SUV_95percentile": suv_95p,
            "organ1_name": organ_info[0][0], "organ1_pct": organ_info[0][1],
            "organ2_name": organ_info[1][0], "organ2_pct": organ_info[1][1],
            "organ3_name": organ_info[2][0], "organ3_pct": organ_info[2][1]
        })

    return lesions_info

def find_label_maps(input_path, pattern, onedir=False):
    if onedir:
        return [os.path.join(input_path, f) for f in os.listdir(input_path) if pattern in f]
    matches = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if pattern in f:
                matches.append(os.path.join(root, f))
    return matches

def main():
    parser = argparse.ArgumentParser(description="Analyze lesion instances in label maps with PET and anatomy data.")
    parser.add_argument("-i", "--input", required=True, help="Input directory or single directory to search.")
    parser.add_argument("--pattern", default="LYM_label.nii.gz", help="Substring to match label maps.")
    parser.add_argument("--anat_pattern", default="_all.nii.gz", help="Substring to match anatomy segmentation.")
    parser.add_argument("--onedir", action="store_true", help="Only search the provided directory, not subfolders.")
    parser.add_argument("--save", action="store_true", help="Save instance label maps.")
    args = parser.parse_args()

    label_maps = find_label_maps(args.input, args.pattern, args.onedir)
    if not label_maps:
        print("No label maps found.")
        return

    for lm in label_maps:
        print(f"Processing: {lm}")
        lesions_info = analyze_lesions(lm, save_instances=args.save,
                                       mask_pattern=args.pattern, anat_pattern=args.anat_pattern)
        
        base = os.path.splitext(os.path.splitext(lm)[0])[0]  # removes .nii.gz
        csv_path = base + "_lesion_stats.csv"

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "lesion_id", "voxel_count", "volume_ml",
                "SUV_max", "SUV_mean", "SUV_95percentile",
                "organ1_name", "organ1_pct", "organ2_name", "organ2_pct", "organ3_name", "organ3_pct"
            ])
            writer.writeheader()
            writer.writerows(lesions_info)

if __name__ == "__main__":
    main()
