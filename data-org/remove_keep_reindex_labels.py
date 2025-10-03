import os
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Manually created loouptable for reindexing SynthSeg label maps
SYNTHSEG_REINDEX_LUT = {
    # original_label : reindexed_label
    0: 0,   # background
    2: 1,   # Left-Cerebral-White-Matter
    3: 2,   # Left-Cerebral-Cortex
    4: 3,   # Left-Lateral-Ventricle
    5: 4,   # Left-Inf-Lat-Vent
    7: 5,   # Left-Cerebellum-White-Matter
    8: 6,   # Left-Cerebellum-Cortex
    10: 7,   # Left-Thalamus
    11: 8,   # Left-Caudate
    12: 9,   # Left-Putamen
    13: 10,  # Left-Pallidum
    14: 11,  # 3rd-Ventricle
    15: 12,  # 4th-Ventricle
    16: 13,  # Brain-Stem
    17: 14,  # Left-Hippocampus
    18: 15,  # Left-Amygdala
    24: 16,  # CSF
    26: 17,  # Left-Accumbens-area
    28: 18,  # Left-VentralDC
    41: 20,  # Right-Cerebral-White-Matter
    42: 21,  # Right-Cerebral-Cortex
    43: 22,  # Right-Lateral-Ventricle
    44: 23,  # Right-Inf-Lat-Vent
    46: 24,  # Right-Cerebellum-White-Matter
    47: 25,  # Right-Cerebellum-Cortex
    49: 26,  # Right-Thalamus
    50: 27,  # Right-Caudate
    51: 28,  # Right-Putamen
    52: 29,  # Right-Pallidum
    53: 30,  # Right-Hippocampus
    54: 31,  # Right-Amygdala
    58: 32,  # Right-Accumbens-area
    60: 33,  # Right-VentralDC
}


def parse_set_to_label(label_list):
    if len(label_list) % 2 != 0:
        raise ValueError("The --set-to-label list must contain an even number of values (pairs of old and new labels).")
    return dict(zip(label_list[::2], label_list[1::2]))


def parse_set_to_one(value_list):
    if len(value_list) == 1 and value_list[0].lower() == "all":
        return "all"
    return [int(v) for v in value_list]

def process_file(args):
    file_path, output_dir, keep_labels, min_label, reindex, set_to_one, reindexing_synthseg, set_to_label, prep4ctx = args

    seg_nii = nib.load(file_path)
    seg = seg_nii.get_fdata()
    seg = np.round(seg).astype(np.int32)

    if keep_labels is not None:
        seg[~np.isin(seg, keep_labels)] = 0

    if min_label is not None:
        seg[seg < min_label] = 0

    if set_to_one == "all":
        seg[seg != 0] = 1
    elif isinstance(set_to_one, list):
        for label in set_to_one:
            seg[seg == label] = 1

    if set_to_label:
        for old_label, new_label in set_to_label.items():
            seg[seg == old_label] = new_label

    if reindexing_synthseg:
        reindexed = np.zeros_like(seg, dtype=np.int32)
        for old_label, new_label in SYNTHSEG_REINDEX_LUT.items():
            if np.any(seg == old_label):
                reindexed[seg == old_label] = new_label
            else:
                print(f"Warning: label {old_label} not found in {os.path.basename(file_path)}")
        seg = reindexed

    elif reindex:
        unique_labels = sorted(list(set(np.unique(seg)) - {0}))
        label_mapping = {old: new for new, old in enumerate(unique_labels, start=1)}
        reindexed = np.zeros_like(seg, dtype=np.int32)
        for old_label, new_label in label_mapping.items():
            reindexed[seg == old_label] = new_label
        seg = reindexed

    seg = seg.astype(np.int32) # Changed from uint8 to int32 to accommodate larger label values

    if reindexing_synthseg:
        filename = os.path.basename(file_path).replace('_0000_synthseg', '')
    else:
        filename = os.path.basename(file_path)

    # === NEW: append _0000 if --prep4ctx is set (always, even if already present) ===
    if prep4ctx:
        if filename.endswith(".nii.gz"):
            name_part = filename[:-7]
            ext = ".nii.gz"
        elif filename.endswith(".nii"):
            name_part = filename[:-4]
            ext = ".nii"
        else:
            name_part, ext = os.path.splitext(filename)

        filename = f"{name_part}_0000{ext}"

    # pick appropriate dtype depending on label range
    if seg.max() <= 255:
        seg = seg.astype(np.uint8)
        new_dtype = np.uint8
    elif seg.max() <= 32767:  # fits in int16
        seg = seg.astype(np.int16)
        new_dtype = np.int16
    else:
        seg = seg.astype(np.int32)
        new_dtype = np.int32

    # make a copy of the header and update datatype
    new_header = seg_nii.header.copy()
    new_header.set_data_dtype(new_dtype)

    output_path = os.path.join(output_dir, filename)
    new_nii = nib.Nifti1Image(seg, affine=seg_nii.affine, header=new_header)
    nib.save(new_nii, output_path)

    return output_path



def main():
    parser = argparse.ArgumentParser(description="Process label maps: filtering, reindexing, and remapping.")
    parser.add_argument("--input", required=True, help="Input directory containing .nii or .nii.gz files.")
    parser.add_argument("--output", required=True, help="Output directory to save processed label maps.")
    parser.add_argument("--keep-labels", type=int, nargs="+", help="Labels to keep (all others set to 0).")
    parser.add_argument("--min-label", type=int, help="Minimum label value to keep (others set to 0).")
    parser.add_argument("--set-to-one", nargs="+", type=str, help="Labels to set to 1, or 'all' to set all non-zero labels.")
    parser.add_argument("--set-to-label", type=int, nargs="+", help="Pairs of old_label new_label (e.g., 13 3 14 3) where, in the given example, label #13 will be reassigned as label #3.")
    parser.add_argument("--reindex", action="store_true", help="Reindex remaining labels to 1..N. Not compatible with --reindexing-synthseg.")
    parser.add_argument("--reindexing-synthseg", action="store_true", help="Use fixed reindexing LUT (SynthSeg-style). Not compatible with --reindex.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--prep4ctx", action="store_true", help="Append '_0000' before extension in output filenames (for context networks).")
    parser.add_argument("--filter-substring", type=str, help="Only process files containing this substring in their filename.")

    args = parser.parse_args()

    if args.reindex and args.reindexing_synthseg:
        parser.error("You cannot use --reindex and --reindexing-synthseg at the same time.")

    # Filter files by substring if provided:
    files = [os.path.join(args.input, f) for f in os.listdir(args.input)
            if (f.endswith(".nii.gz") or f.endswith(".nii")) and
                (args.filter_substring is None or args.filter_substring in f)]

    if args.filter_substring is None:
        print("No filter_substring provided: processing all .nii/.nii.gz files.")
    else:
        print(f"File filter substring='{args.filter_substring}' provided. Processing {len(files)} file(s) matching the substring.")
        print("Files to be processed:")
        for f in files:
            print(f"  {os.path.basename(f)}")

    set_to_one = parse_set_to_one(args.set_to_one) if args.set_to_one else None
    set_to_label = parse_set_to_label(args.set_to_label) if args.set_to_label else None

    os.makedirs(args.output, exist_ok=True)

    print(f"Processing {len(files)} files with {args.num_workers} workers...")

    task_args = [
        (f, args.output, args.keep_labels, args.min_label, args.reindex,
         set_to_one, args.reindexing_synthseg, set_to_label, args.prep4ctx)
        for f in files
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(process_file, task_args), total=len(files), desc="Processing"))

    print("All files processed.")


if __name__ == "__main__":
    main()
