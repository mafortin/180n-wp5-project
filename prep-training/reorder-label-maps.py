import os
import json
import argparse
import nibabel as nib
import numpy as np

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Process and rename label maps in a directory.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Path to the input directory containing label maps.")
    parser.add_argument('--output_dir', type=str, required=False,
                        help="Path to the output directory to save processed label maps.")
    parser.add_argument('--mapping_dir', type=str, help="Path to the directory to save label mappings. If not provided, mappings will not be saved.")
    parser.add_argument('--json_name', type=str, required=False, default="aseg2linear", help="Suffix for the JSON label conversion files.")
    parser.add_argument('--all2one', action='store_true', help="If set, all labels except 0 are set to 1.")
    args = parser.parse_args()

    # Derive output_dir and mapping_dir from input_dir if they are not provided
    if not args.output_dir:
        args.output_dir = args.input_dir + '-reord'
    if not args.mapping_dir:
        args.mapping_dir = os.path.join(args.input_dir, 'labels_mapping')
    if args.all2one:
        args.json_name = "all2one"

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mapping directory: {args.mapping_dir}")
    print(f"JSON name prefix: {args.json_name}")
    print(f"All-to-one mode: {args.all2one}")

    process_directory(args.input_dir, args.output_dir, args.json_name, args.mapping_dir, args.all2one)
    print("Done processing label values and saving label mappings.")

def save_labels(mapping, json_name, mapping_dir, file_basename):
    # Convert keys to standard Python integers
    mapping = {int(k): v for k, v in mapping.items()}

    # Create a unique filename for each mapping
    mapping_file_path = os.path.join(mapping_dir, f'{file_basename}_{json_name}.json')
    with open(mapping_file_path, 'w') as mapping_file:
        json.dump(mapping, mapping_file)
    print(f"Labels conversion saved to {mapping_file_path}")

def process_label_map(file_path, output_dir, json_name, mapping_dir=None, all2one=False):

    # Load the label map file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Ensure the data is handled as integer
    data = np.round(data).astype(np.int32)  # Round and convert to integer

    # Debug: print the shape and data type of the loaded image
    print(f"Loaded {file_path}, shape: {data.shape}, dtype: {data.dtype}")

    if all2one:
        # Get unique labels and create a mapping
        unique_labels = list(map(int, np.unique(data)))  # Convert to standard Python integers
        label_map = {old: (1 if old != 0 else 0) for old in unique_labels}
        # If all2one is enabled, map all labels except 0 to 1
        new_data = np.where(data > 0, 1, 0)
        print(f"All-to-one mode enabled. All labels except 0 are set to 1.")
    else:
        # Get unique labels and create a mapping
        unique_labels = list(map(int, np.unique(data)))  # Convert to standard Python integers
        label_map = {old: new for new, old in enumerate(unique_labels)}

        # Debug: print the unique labels and the label map
        print(f"Unique labels in {file_path}: {unique_labels}")

        # Map the old labels to new labels
        new_data = np.copy(data)
        for old_label, new_label in label_map.items():
            new_data[data == old_label] = new_label

    # Extract the base name for saving mappings
    file_basename = os.path.basename(file_path).replace('.nii.gz', '')

    # Optionally save the mapping
    if mapping_dir:
        save_labels(label_map, json_name, mapping_dir, file_basename)

    # Save the new label map file
    new_img = nib.Nifti1Image(new_data, img.affine, img.header)
    new_file_path = os.path.join(output_dir, os.path.basename(file_path))
    nib.save(new_img, new_file_path)
    print(f"Processed {file_path} -> {new_file_path}")

def process_directory(input_dir, output_dir, json_name, mapping_dir=None, all2one=False):
    # Ensure the output and mapping directories exist
    os.makedirs(output_dir, exist_ok=True)
    if mapping_dir:
        os.makedirs(mapping_dir, exist_ok=True)

    # Process each .nii.gz file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_dir, filename)
            process_label_map(file_path, output_dir, json_name, mapping_dir, all2one)

if __name__ == "__main__":
    main()

