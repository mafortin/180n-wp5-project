import os
import json
import argparse

def extract_sub_id(filename, start_substring='sub', end_substring='_'):
    start = filename.find(start_substring)
    if start == -1:
        return None
    start += len(start_substring)  # Move start to the end of the start_substring
    end = filename.find(end_substring, start)
    if end == -1:
        return None
    return filename[start:end]

def rename_files_in_subfolder(subfolder, output_folder=None, start_substring='sub', end_substring='_', segs=False, mods=None, labels=None, change_ids=False):
    # If output folder is not provided, use the same as the input folder
    if output_folder is None:
        output_folder = subfolder

    # Create output folders if they don't exist
    imagesTr_folder = os.path.join(output_folder, 'imagesTr')
    labelsTr_folder = os.path.join(output_folder, 'labelsTr')
    os.makedirs(imagesTr_folder, exist_ok=True)
    os.makedirs(labelsTr_folder, exist_ok=True)

    # Determine the path for the JSON file in the parent directory of the input folder
    parent_dir = os.path.abspath(os.path.join(subfolder, os.pardir))
    json_file_path = os.path.join(parent_dir, 'file_correspondence.json')

    # Check if the JSON file already exists
    if os.path.exists(json_file_path):
        # Read the existing JSON file
        with open(json_file_path, 'r') as jsonfile:
            rename_mapping = json.load(jsonfile)
    else:
        rename_mapping = {}

    # Iterate over each file in the subfolder
    for file_name in os.listdir(subfolder):
        old_file_path = os.path.join(subfolder, file_name)
        old_id = extract_sub_id(file_name, start_substring, end_substring)
        if old_id is None:
            continue
        if old_id in rename_mapping:
            new_id = rename_mapping[old_id]
        else:
            new_id = old_id if not change_ids else f"image_{len(rename_mapping):04d}"
            rename_mapping[old_id] = new_id
            with open(json_file_path, 'w') as jsonfile:
                json.dump(rename_mapping, jsonfile, indent=4)

        # Determine the new file name and path based on modality or segmentation
        if mods:
            for i, mod in enumerate(mods):
                if mod in file_name:
                    new_file_name = f"image_{new_id}_{i:04d}.nii.gz"
                    new_file_path = os.path.join(imagesTr_folder, new_file_name)
                    os.rename(old_file_path, new_file_path)
                    break
        elif labels and any(label in file_name for label in labels):
            new_file_name = f"image_{new_id}.nii.gz"
            new_file_path = os.path.join(labelsTr_folder, new_file_name)
            os.rename(old_file_path, new_file_path)
        else:
            new_file_name = f"image_{new_id}_0000.nii.gz"
            new_file_path = os.path.join(imagesTr_folder, new_file_name)
            os.rename(old_file_path, new_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename files in a subfolder and save the correspondence to a JSON file.')
    parser.add_argument('--input_folder', type=str, help='Path to the subfolder containing files to rename', required=True)
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to save the renamed files and JSON file')
    parser.add_argument('--start_substring', type=str, default='sub', help='Substring to start extracting the sub_id')
    parser.add_argument('--end_substring', type=str, default='_', help='Substring to end extracting the sub_id')
    parser.add_argument('--segs', action='store_true', help='Flag indicating that the files to rename are label maps and should not have the _0000 suffix')
    parser.add_argument('--mod', nargs='+', help='List of substrings to identify each modality')
    parser.add_argument('--seg_id', nargs='+', help='List of substrings to identify segmentations/label maps')
    parser.add_argument('--change_ids', action='store_true', help='Flag indicating whether to change subject IDs')
    args = parser.parse_args()

    rename_files_in_subfolder(args.input_folder, args.output_folder, args.start_substring, args.end_substring, args.segs, args.mod, args.labels, args.change_ids)

