#!/usr/bin/env python3

import os
import argparse
import SimpleITK as sitk

def convert_nrrd_to_nii(input_root, output_root):
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.nrrd'):
                input_path = os.path.join(root, file)

                # Build output path
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_filename = os.path.splitext(file)[0] + '.nii.gz'
                output_path = os.path.join(output_dir, output_filename)

                try:
                    img = sitk.ReadImage(input_path)
                    sitk.WriteImage(img, output_path)
                    print(f"Converted: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Failed to convert {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .nrrd files to .nii.gz recursively.")
    parser.add_argument("-i", "--input", required=True, help="Path to input folder containing .nrrd files.")
    parser.add_argument("-o", "--output", help="Path to output folder. Defaults to input folder if not provided.")

    args = parser.parse_args()
    output_root = args.output if args.output else args.input

    convert_nrrd_to_nii(args.input, output_root)

if __name__ == "__main__":
    main()
