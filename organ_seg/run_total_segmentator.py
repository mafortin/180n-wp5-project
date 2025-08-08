#!/usr/bin/env python3

import subprocess
import sys
import argparse
import os
import glob

def run_total_segmentator(input_path, output_path, task='total_mr', one=True, labels=None):
    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_path,
        "--task", task
    ]

    if one:
        cmd.append("-ml")

    if labels is not None:
        cmd.append("--roi_subset")
        cmd.extend(labels)

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

def get_output_filename(input_file, output_dir, suffix, sub_id_override=None):
    if sub_id_override:
        name = sub_id_override
        ext = '.nii.gz'
    else:
        base = os.path.basename(input_file)
        if base.endswith('.nii.gz'):
            name = base[:-7]
            ext = '.nii.gz'
        elif base.endswith('.nii'):
            name = base[:-4]
            ext = '.nii'
        else:
            name, ext = os.path.splitext(base)
    
    return os.path.join(output_dir, f"{name}{suffix}{ext}")

def find_nested_nii_files(input_dir):
    nii_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                sub_id = os.path.basename(os.path.dirname(full_path))
                nii_files.append((full_path, sub_id))
    return nii_files

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run TotalSegmentator on all NIfTI files in a directory or subdirectories.")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing all images or subdirectories with images to be segmented.")
    parser.add_argument("-o", "--output", required=True, help="Output directory where all the label maps will be saved.")
    parser.add_argument("-t", "--task", default="total_mr", help="Task to run for the main segmentation (default: total_mr).")
    parser.add_argument("--one", action="store_true", help="Create one final output label map with all labels (instead of N label maps for all labels individually).")
    parser.add_argument("-l", "--labels", nargs="+", default=None, help="List of labels to segment.")
    parser.add_argument("-s", "--suffix", default="_oseg", help="Suffix to add before the file extension for output files. If you don't do anything special, keep the default.")
    parser.add_argument("--subdirs", action="store_true", help="Search for images in subdirectories. Uses subdirectory name as subject ID.")
    parser.add_argument("--filter", type=str, help="Only segment files containing this substring in their filename.")
    parser.add_argument("--no_body_mr", action="store_true", help="Do not run the additional body_mr segmentation.")

    args = parser.parse_args()

    if args.labels is not None:
        print(f"Labels to be segmented: {args.labels}")
    else:
        args.labels = None
        print("Labels to be segmented: All")

    if args.one:
        print("Creating one final output label map with all labels.")
    else:
        print("Creating one output label map per structure segmented.")
        print("Note: This is currently not supported with the current implementation nor advised due to the creation of 50 files per segmented image.")

    os.makedirs(args.output, exist_ok=True)

    def filename_matches_filter(filename, filter_str):
        return filter_str in os.path.basename(filename) if filter_str else True

    if args.subdirs:
        all_files = find_nested_nii_files(args.input)
        nii_files = [(f, sub_id) for f, sub_id in all_files if filename_matches_filter(f, args.filter)]
    else:
        paths = glob.glob(os.path.join(args.input, "*.nii")) + glob.glob(os.path.join(args.input, "*.nii.gz"))
        nii_files = [(path, None) for path in paths if filename_matches_filter(path, args.filter)]

    if not nii_files:
        print("No .nii or .nii.gz files found.")
        sys.exit(1)

    print(f"Found {len(nii_files)} NIfTI file(s). Segmenting...")

    for nii_file, sub_id in nii_files:
        
        if sub_id is not None and args.subdirs:
            output_subdir = os.path.join(args.output, sub_id)
            os.makedirs(output_subdir, exist_ok=True)
            output_file = get_output_filename(nii_file, output_subdir, args.suffix)
        else:
            output_file = get_output_filename(nii_file, args.output, args.suffix)

        # Run main task (total_mr by default)
        print(f"Processing {nii_file} -> {output_file} with task {args.task}")
        run_total_segmentator(nii_file, output_file, args.task, args.one, args.labels)

        # Run additional body_mr unless disabled
        if not args.no_body_mr:
            body_suffix = "_body"
            if sub_id is not None and args.subdirs:
                output_file_body = get_output_filename(nii_file, output_subdir, body_suffix)
            else:
                output_file_body = get_output_filename(nii_file, args.output, body_suffix)
            print(f"Processing {nii_file} -> {output_file_body} with task body_mr")
            run_total_segmentator(nii_file, output_file_body, "body_mr", args.one, args.labels)

    print("TotalSegmentator run completed successfully.")
