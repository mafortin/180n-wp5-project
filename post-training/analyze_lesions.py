#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
import numpy as np
import csv
from scipy.ndimage import label, center_of_mass

# ──────────────────────────────────────────────────────────────────────────────
# Organ label mapping with anatomical metadata:
# Format: label_id: 
# (organ_name, above_diaphragm_flag [Yes=1/No=0/both=None], laterality [Not applicable=NA], lymph node site associated)
ALL_MR_LABELS_INFO = {
    1:  ("spleen", 0, "left", ["spleen"]),
    2:  ("kidney_right", 0, "right", ["paraaortic"]),
    3:  ("kidney_left", 0, "left", ["paraaortic"]),
    4:  ("gallbladder", 0, "right", ["mesenteric"]),
    5:  ("liver", 0, "right", ["extranodal"]),
    6:  ("stomach", 0, "left", ["mesenteric"]),
    7:  ("pancreas", 0, "left", ["mesenteric"]),
    8:  ("adrenal_gland_right", 0, "right", ["paraaortic"]),
    9:  ("adrenal_gland_left", 0, "left", ["paraaortic"]),
    10: ("lung_left", 1, "left", ["hilar"]),
    11: ("lung_right", 1, "right", ["hilar"]),
    12: ("esophagus", 1, "NA", ["mediastinal"]),
    13: ("small_bowel", 0, "NA", ["mesenteric"]),
    14: ("duodenum", 0, "NA", ["mesenteric"]),
    15: ("colon", 0, "NA", ["mesenteric"]),
    16: ("urinary_bladder", 0, "NA", ["iliac", "femoral/inguinal"]),
    17: ("prostate", 0, "NA", ["iliac", "femoral/inguinal"]),
    18: ("sacrum", 0, "NA", ["iliac"]),
    19: ("vertebrae", None, "NA", ["extranodal"]),
    20: ("intervertebral_discs", None, "NA", ["extranodal"]),
    21: ("spinal_cord", None, "NA", ["extranodal"]),
    22: ("heart", 1, "NA", ["mediastinal"]),
    23: ("aorta", 1, "left", ["mediastinal", "paraaortic"]),
    24: ("inferior_vena_cava", 0, "right", ["paraaortic"]),
    25: ("portal_vein_and_splenic_vein", 0, "NA", ["mesenteric"]),
    26: ("iliac_artery_left", 0, "left", ["iliac"]),
    27: ("iliac_artery_right", 0, "right", ["iliac"]),
    28: ("iliac_vena_left", 0, "left", ["iliac"]),
    29: ("iliac_vena_right", 0, "right", ["iliac"]),
    30: ("humerus_left", 1, "left", ["axillary", "pectoral"]),
    31: ("humerus_right", 1, "right", ["axillary_right", "pectoral"]),
    32: ("scapula_left", 1, "left", ["axillary", "pectoral"]),
    33: ("scapula_right", 1, "right", ["axillary", "pectoral"]),
    34: ("clavicula_left", 1, "left", ["supraclavicular", "infraclavicular"]),
    35: ("clavicula_right", 1, "right", ["supraclavicular", "infraclavicular"]),
    36: ("femur_left", 0, "left", ["femoral/inguinal"]),
    37: ("femur_right", 0, "right", ["femoral/inguinal"]),
    38: ("hip_left", 0, "left", ["iliac"]),
    39: ("hip_right", 0, "right", ["iliac"]),
    40: ("gluteus_maximus_left", 0, "left", ["iliac"]),
    41: ("gluteus_maximus_right", 0, "right", ["iliac"]),
    42: ("gluteus_medius_left", 0, "left", ["extranodal"]),
    43: ("gluteus_medius_right", 0, "right", ["extranodal"]),
    44: ("gluteus_minimus_left", 0, "left", ["extranodal"]),
    45: ("gluteus_minimus_right", 0, "right", ["extranodal"]),
    46: ("autochthon_left", None, "left", ["extranodal"]),
    47: ("autochthon_right", None, "right", ["extranodal"]),
    48: ("iliopsoas_left", 0, "left", ["iliac"]),
    49: ("iliopsoas_right", 0, "right", ["iliac"]),
    50: ("brain", 1, "NA", ["cervical/occipital/preauricular/waldeyer's ring"]),
    101: ("trunc", None, "NA", []),
    103: ("head&neck", 1, "NA", ["cervical/occipital/preauricular/waldeyer's ring"]),
    104: ("arm_left", None, "left", ["axillary"]),
    105: ("arm_right", None, "right", ["axillary"]),
    106: ("leg_left", 1, "left", ["femoral/inguinal"]),
    107: ("leg_right", 1, "right", ["femoral/inguinal"]),
}


# Simple id->name mapping
ALL_MR_LABELS = {k: v[0] for k, v in ALL_MR_LABELS_INFO.items()}



def get_pet_path(mask_path):
    base_dir = os.path.dirname(mask_path)
    fname_noext = os.path.splitext(os.path.splitext(os.path.basename(mask_path))[0])[0]
    if "_LYM" in fname_noext:
        subj_prefix = fname_noext.split("_LYM")[0]
        return os.path.join(base_dir, f"{subj_prefix}_LYM.nii.gz")
    else:
        return os.path.join(base_dir, fname_noext + "_LYM.nii.gz")

def classify_lesion_position(organ_names):
    """
    Classifies lesion position as above/below diaphragm and left/right
    based on organ overlap.
    Returns:
        above_diaphragm: 1 (above), 0 (below), None (spans or unknown)
        laterality: "left", "right", "NA" (midline), or "unknown"
    """
    fallback_organs = [name for name in organ_names if name not in ("trunc", "None")]
    if not fallback_organs:
        return None, "unknown"

    for organ in fallback_organs:
        label_id = next((k for k, v in ALL_MR_LABELS.items() if v == organ), None)
        if label_id is not None:
            above_flag, side = ALL_MR_LABELS_INFO[label_id][1], ALL_MR_LABELS_INFO[label_id][2]
            return above_flag, side
    return None, "unknown"



def lesion_centroid(mask):
    return np.array(center_of_mass(mask))


def classify_lesion(organ_ids, lesion_mask, organ_data):
    # Handle empty list
    if not organ_ids:
        return "unknown"

    # Compute lesion centroid
    lesion_c = lesion_centroid(lesion_mask)

    # Get lymph node sites for each overlapping organ
    possible_sites = []
    for oid in organ_ids:
        possible_sites.extend(ALL_MR_LABELS_INFO[oid][3])
    possible_sites = list(set(possible_sites))

    # If lesion overlaps only one type of site → assign directly
    if len(possible_sites) == 1:
        return possible_sites[0]

    # Special handling for trunk-primary lesions near clavicles
    if organ_ids and organ_ids[0] == 101:
        lesion_c = lesion_centroid(lesion_mask)

        # Get centroids for head and lungs to filter Z range
        head_mask = organ_data == 103
        lung_mask = (organ_data == 10) | (organ_data == 11)
        if np.any(head_mask) and np.any(lung_mask):
            head_c = lesion_centroid(head_mask)
            lung_c = lesion_centroid(lung_mask)

            # Superior-inferior axis index in centroid array
            z_idx = 2

            # Lesion must be below head and above lungs along Z
            if lesion_c[z_idx] < head_c[z_idx] and lesion_c[z_idx] > lung_c[z_idx]:
                # Get centroids of clavicles
                clav_centroids = {}
                for cid in (34, 35):
                    if np.any(organ_data == cid):
                        clav_centroids[cid] = lesion_centroid(organ_data == cid)

                if clav_centroids:
                    # Find closest clavicle centroid in 3D space
                    closest_cid = min(
                        clav_centroids,
                        key=lambda cid: np.linalg.norm(lesion_c - clav_centroids[cid])
                    )
                    closest_clav_c = clav_centroids[closest_cid]

                    if lesion_c[z_idx] > closest_clav_c[z_idx]:
                        return "supraclavicular"
                    else:
                        return "infraclavicular"



def analyze_lesions(label_map_path, lesion_pattern="LYM_label.nii.gz",
                    anat_pattern="_all.nii.gz", topn=5):

    # Find anatomy segmentation
    print("Searching for anatomy/organ segmentation...")
    anat_path = None
    for f in os.listdir(os.path.dirname(label_map_path)):
        if anat_pattern in f:
            anat_path = os.path.join(os.path.dirname(label_map_path), f)
            break
    
    # Find PET image
    pet_path = pet_path = get_pet_path(label_map_path)

    # Load and check shapes
    label_shape = nib.load(label_map_path).shape
    anat_shape = nib.load(anat_path).shape
    pet_shape = nib.load(pet_path).shape

    if label_shape != anat_shape or label_shape != pet_shape:
        raise ValueError(
            f"Shape mismatch detected:\n"
            f" - Lesion label map: {label_shape}\n"
            f" - Anatomy segmentation: {anat_shape}\n"
            f" - PET image: {pet_shape}\n\n"
            "Please run 'resample_multi_modality_images.py' to align image dimensions before running this script."
        )

    print(f"\nLoading lesion mask: {label_map_path}")
    img = nib.load(label_map_path)
    data = img.get_fdata().astype(np.uint8)

    print("Performing instance segmentation...")
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_array, num_features = label(data, structure=struct)
    print(f"Found {num_features} lesion instances.")

    base = os.path.splitext(os.path.splitext(label_map_path)[0])[0]
    inst_path = base + "_inst.nii.gz"
    print(f"Saving instance segmentation to: {inst_path}")
    nib.save(nib.Nifti1Image(labeled_array, img.affine, img.header), inst_path)

    # Load PET image
    if os.path.exists(pet_path):
        print(f"Loading PET image: {pet_path}")
        pet_img = nib.load(pet_path)
        pet_data = pet_img.get_fdata()
    else:
        print(f"WARNING: PET image not found for {label_map_path}. SUV metrics will be skipped.")
        pet_data = None


    anat_data = None
    liver_suv95 = None
    aorta_suv95 = None

    if anat_path and os.path.exists(anat_path):
        print(f"Loading anatomy/organ segmentation: {anat_path}")
        anat_img = nib.load(anat_path)
        anat_data = anat_img.get_fdata().astype(int)

        if pet_data is not None:
            liver_mask = anat_data == 5   # liver label ID
            aorta_mask = anat_data == 23  # aorta label ID

            if np.any(liver_mask):
                liver_suv95 = float(np.percentile(pet_data[liver_mask], 95))
            if np.any(aorta_mask):
                aorta_suv95 = float(np.percentile(pet_data[aorta_mask], 95))
    else:
        print(f"WARNING: Anatomy segmentation not found for {label_map_path}")

    voxel_volume_ml = np.prod(img.header.get_zooms()) / 1000.0
    lesions_info = []

    print("Analyzing individual lesions...")
    for lesion_id in range(1, num_features + 1):
        lesion_mask = (labeled_array == lesion_id)
        voxel_count = np.sum(lesion_mask)
        volume_ml = voxel_count * voxel_volume_ml

        # PET-based metrics
        if pet_data is not None:
            suv_values = pet_data[lesion_mask]
            if suv_values.size > 0:
                suv_max = float(np.max(suv_values))
                suv_mean = float(np.mean(suv_values))
                suv_95p = float(np.percentile(suv_values, 95))
            else:
                suv_max = suv_mean = suv_95p = 0.0
        else:
            suv_max = suv_mean = suv_95p = 0.0

        # Organ overlap
        organ_info = [("None", 0.0), ("None", 0.0)]
        chosen_main_organ = None
        if anat_data is not None:
            lesion_organs_raw, counts_raw = np.unique(anat_data[lesion_mask], return_counts=True)
            mask = lesion_organs_raw != 0
            lesion_organs = lesion_organs_raw[mask]
            counts = counts_raw[mask]

            if len(lesion_organs) > 0:
                total_vox = np.sum(counts)
                percents = (counts / total_vox) * 100
                sort_idx = np.argsort(percents)[::-1]
                lesion_organs = lesion_organs[sort_idx]
                percents = percents[sort_idx]

                top_n = min(2, len(lesion_organs))
                organ_info = []
                for i in range(top_n):
                    label_id = int(round(lesion_organs[i]))
                    organ_name = ALL_MR_LABELS.get(label_id, f"unknown_label_{label_id}")
                    organ_info.append((organ_name, round(float(percents[i]), 2)))
                while len(organ_info) < 2:
                    organ_info.append(("None", 0.0))

                # New selection rule: skip trunks unless no other choice
                for oid in lesion_organs:
                    if oid != 101:  # not trunc
                        chosen_main_organ = ALL_MR_LABELS.get(int(oid), f"unknown_label_{oid}")
                        break
                if chosen_main_organ is None:
                    chosen_main_organ = "unknown"

        # Position & laterality (proxy)
        usable_organs = [chosen_main_organ] if chosen_main_organ not in ("None", "unknown") else []
        above_diaphragm, side = classify_lesion_position(usable_organs or ["None"])

        # Lymph node region
        # Convert organ names to IDs
        organ_ids = [oid for oid, name in ALL_MR_LABELS.items() if name in [o for o, _ in organ_info]]

        ln_region = classify_lesion(
            organ_ids=organ_ids,
            lesion_mask=lesion_mask,
            organ_data=anat_data
        )


        lesions_info.append({
            "lesion_id": lesion_id,
            "voxel_count": int(voxel_count),
            "volume_ml": volume_ml,
            "SUV_max": suv_max,
            "SUV_mean": suv_mean,
            "SUV_95percentile": suv_95p,
            "organ1_name": organ_info[0][0], "organ1_pct": organ_info[0][1],
            "organ2_name": organ_info[1][0], "organ2_pct": organ_info[1][1],
            "above_diaphragm": above_diaphragm,
            "laterality": side,
            "lymph_node_region": ln_region,
            "deauville_score": None
        })


    # Deauville score
    for lesion in lesions_info:
        lesion["deauville_score"] = None  # default to None

    highest_lesion = None
    if pet_data is not None and liver_suv95 is not None and aorta_suv95 is not None:
        highest_lesion = max(lesions_info, key=lambda x: x["SUV_95percentile"])
        suv95_top = highest_lesion["SUV_95percentile"]

        if suv95_top <= aorta_suv95:
            score = 2
        elif suv95_top <= liver_suv95:
            score = 3
        elif suv95_top <= liver_suv95 * 1.5:
            score = 4
        else:
            score = 5

        highest_lesion["deauville_score"] = score

    # Summary output
    print(f"\nTop {topn} largest lesions (out of {len(lesions_info)} total):")
    top_n = sorted(lesions_info, key=lambda x: x["volume_ml"], reverse=True)[:topn]
    for lesion in top_n:
        organs = []
        for i in range(1, 3):
            name = lesion[f"organ{i}_name"]
            pct = lesion[f"organ{i}_pct"]
            if name != "None" and pct > 1.0:
                organs.append(f"{name} ({pct}%)")
        organs_str = ", ".join(organs) if organs else "No significant overlap"

        ln_str = lesion.get("lymph_node_region", "unknown")
        if lesion.get("ln_region_confidence", 0.0) > 0:
            ln_str += f" (conf {lesion['ln_region_confidence']:.2f})"

        print(
            f"- Lesion {lesion['lesion_id']}: "
            f"Volume = {lesion['volume_ml']:.2f} mL, "
            f"Organs = {organs_str}, "
            f"LN region = {ln_str}"
        )

    if highest_lesion is not None:
        print(f"\nDeauville Score for highest uptake lesion (Lesion {highest_lesion['lesion_id']}):")
        print(f"  SUV_95% (lesion) = {highest_lesion['SUV_95percentile']:.0f}")
        print(f"  SUV_95% (aorta)  = {aorta_suv95:.0f}")
        print(f"  SUV_95% (liver)  = {liver_suv95:.0f}")
        print(f"Deauville Score: {highest_lesion['deauville_score']} (only relevant if interim or final visit)")

    return lesions_info


def find_label_maps(input_path, pattern, onedir=False):
    print(f"Searching for label maps in: {input_path}")
    if onedir:
        return [os.path.join(input_path, f) for f in os.listdir(input_path) if pattern in f]
    matches = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if pattern in f:
                matches.append(os.path.join(root, f))
    print(f"Found {len(matches)} label maps to process inside .")
    return matches


def main():
    parser = argparse.ArgumentParser(description="Analyze lesion instances in label maps with PET and anatomy data, and classify upper-body lymph-node regions.")
    parser.add_argument("-i", "--input", required=True, help="Input directory or single directory to search.")
    parser.add_argument("--lesion_pattern", default="LYM_label.nii.gz", help="Substring to match the lesion label maps.")
    parser.add_argument("--anat_pattern", default="_all.nii.gz", help="Substring to match anatomy segmentation.")
    parser.add_argument("--onedir", action="store_true", help="Only search the provided directory, not subfolders.")
    parser.add_argument("--topn", type=int, default=5, help="Number of largest lesions to print summary for.")
    args = parser.parse_args()

    label_maps = find_label_maps(args.input, args.lesion_pattern, args.onedir)
    if not label_maps:
        print("No label maps found.")
        return

    for lm in label_maps:
        if args.onedir:
            subject_id = os.path.basename(lm).split("_")[0]
        else:
            subject_id = os.path.basename(os.path.dirname(lm))

        print(f"\n--- Processing subject: {subject_id} ---")


        lesions_info = analyze_lesions(lm,
                                       lesion_pattern=args.lesion_pattern,
                                       anat_pattern=args.anat_pattern,
                                       topn=args.topn)

        base = os.path.splitext(os.path.splitext(lm)[0])[0]
        csv_path = base + "_lesion_stats.csv"
        print(f"Saving lesion statistics to: {csv_path}")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "lesion_id", "voxel_count", "volume_ml",
                "SUV_max", "SUV_mean", "SUV_95percentile",
                "organ1_name", "organ1_pct", "organ2_name", "organ2_pct", "organ3_name", "organ3_pct",
                "above_diaphragm", "laterality",
                "lymph_node_region",
                "deauville_score"
            ])
            writer.writeheader()
            writer.writerows(lesions_info)

        print(f"Finished processing subject: {subject_id}")


if __name__ == "__main__":
    main()