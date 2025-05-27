import subprocess
import sys
import argparse
import os
import glob

def run_total_segmentator(input_path, output_path, one=True, labels=None, verbose=True):
    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_path
    ]

    if verbose:
        cmd.append("-v")

    if one:
        cmd.append("-ml")

    if labels is not None:
        cmd.append("-rs")
        cmd.extend(labels)

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

def get_output_filename(input_file, output_dir, suffix):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TotalSegmentator on all NIfTI files in a directory.")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing all images to be segmented.")
    parser.add_argument("-o", "--output", required=True, help="Output directory where all the label maps will be saved.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--one", action="store_true", help="Create one final output label map with all labels. If not set, there will be one output label map per structure segmented.")
    parser.add_argument("-l", "--labels", nargs="+", default=None, help="List of labels to segment. If not set, a pre-defined subset of the relevant labels segmentable by the `total_mr` task will be segmented (the user can see the list at the end of this script).")
    parser.add_argument("-s", "--suffix", default="_oseg", help="Suffix to add before the file extension for output files (default: _oseg)")

    args = parser.parse_args()

    if args.labels is not None:
        print(f"Labels to be segmented: {args.labels}")
    else:
        args.labels = [
            "liver", "spleen", "kidney_right", "kidney_left", "pancreas", "gallbladder", "stomach", "urinary_bladder", "prostate", "inferior_vena_cava",
            "heart", "aorta", "lung_left", "lung_right", "femur_left", "femur_right", "clavicle_left", "clavicle_right", "hip_left", "hip_right",
            "esophagus", "brain", "spinal_cord", "prostate", "iliac_artery_left", "iliac_artery_right", "iliac_vein_left", "iliac_vein_right",
        ]

    if args.one:
        print("Creating one final output label map with all labels.")
    else:
        print("Creating one output label map per structure segmented.")

    os.makedirs(args.output, exist_ok=True)
    nii_files = glob.glob(os.path.join(args.input, "*.nii")) + glob.glob(os.path.join(args.input, "*.nii.gz"))

    if not nii_files:
        print("No .nii or .nii.gz files found in the input directory.")
        sys.exit(1)

    for nii_file in nii_files:
        output_file = get_output_filename(nii_file, args.output, args.suffix)
        print(f"Processing {nii_file} -> {output_file}")
        run_total_segmentator(nii_file, output_file, args.one, args.labels, args.verbose)

    print("TotalSegmentator run completed successfully.")
