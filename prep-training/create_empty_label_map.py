import os
import argparse
import nibabel as nib
import numpy as np

#!/usr/bin/env python3


def create_empty_label_map(input_dir, reference_filename, output_filename):
    # Loop through all subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):


        for subdir in dirs:
            
            # Extract the sub_id from the subfolder name
            sub_id = subdir
            subdir_path = os.path.join(root, subdir)
            
            # Update the output filename to include the sub_id
            output_file_path = os.path.join(subdir_path, f"{sub_id}{output_filename}")

            if not os.path.exists(output_file_path):
                # Check if the reference file exists
                reference_file_path = os.path.join(subdir_path,  f"{sub_id}{reference_filename}")
                if os.path.exists(reference_file_path):
                    reference_image = nib.load(reference_file_path)
                    reference_data = reference_image.get_fdata()
                    affine = reference_image.affine
                    # Create a 3D matrix of zeros with the same shape as the reference
                    empty_data = np.zeros_like(reference_data, dtype=np.int16)  # Use integer type
                    
                    # Create a new NIfTI image with the empty data
                    empty_image = nib.Nifti1Image(empty_data, affine)
                    
                    # Save the new image to the output file
                    nib.save(empty_image, output_file_path)
                    print(f"Created empty label map: {output_file_path}")
                else:
                    print(f"Reference file not found: {reference_file_path}")
            else:
                print(f"Output file already exists: {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Create empty label maps for missing files.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing subfolders.")
    parser.add_argument("-r", "--reference_filename", required=True, help="Filename of the reference .nii.gz file.")
    parser.add_argument("-o", "--output_filename", required=True, help="Filename of the output .nii.gz file.")
    
    args = parser.parse_args()
    
    create_empty_label_map(args.input_dir, args.reference_filename, args.output_filename)

if __name__ == "__main__":
    main()