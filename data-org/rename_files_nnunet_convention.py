import os
import json
import argparse
import shutil

def rename_files_in_subfolder(subfolder, sub_id, imagesTr_folder, labelsTr_folder, json_file_path, segs=False, mods=None, seg_id=None, change_ids=False):
    
    # Check if the JSON file already exists
    if os.path.exists(json_file_path):
        # Read the existing JSON file
        with open(json_file_path, 'r') as jsonfile:
            rename_mapping = json.load(jsonfile)
    else:
        rename_mapping = {}

    # Create the rename mapping if not already present
    if sub_id not in rename_mapping:
        new_id = sub_id if not change_ids else f"image_{len(rename_mapping):04d}"
        rename_mapping[sub_id] = new_id

    # Save the rename mapping to the JSON file if change_ids is set
    if change_ids:
        with open(json_file_path, 'w') as jsonfile:
            json.dump(rename_mapping, jsonfile, indent=4)

    new_id = rename_mapping[sub_id]

    # Iterate over each file in the subfolder and rename
    for file_name in os.listdir(subfolder):
        old_file_path = os.path.join(subfolder, file_name)

        # Determine the new file name and path based on modality or segmentation
        if mods:
            for i, mod in enumerate(mods):
                if mod in file_name:
                    new_file_name = f"image_{new_id}_{i:04d}.nii.gz"
                    new_file_path = os.path.join(imagesTr_folder, new_file_name)
                    shutil.copy(old_file_path, new_file_path)
                    print(f'Copied: {old_file_path} -> {new_file_path}')
        elif seg_id and any(label in file_name for label in seg_id):
            new_file_name = f"image_{new_id}.nii.gz"
            new_file_path = os.path.join(labelsTr_folder, new_file_name)
            shutil.copy(old_file_path, new_file_path)
            print(f'Copied: {old_file_path} -> {new_file_path}')
        else:
            new_file_name = f"image_{new_id}_0000.nii.gz"
            new_file_path = os.path.join(imagesTr_folder, new_file_name)
            shutil.copy(old_file_path, new_file_path)
            print(f'Copied: {old_file_path} -> {new_file_path}')

def process_main_folder(main_folder, output_folder=None, segs=False, mods=None, seg_id=None, change_ids=False):
    # If output folder is not provided, use the same as the input folder
    if output_folder is None:
        output_folder = main_folder

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create output folders if they don't exist
    imagesTr_folder = os.path.join(output_folder, 'imagesTr')
    labelsTr_folder = os.path.join(output_folder, 'labelsTr')
    os.makedirs(imagesTr_folder, exist_ok=True)
    os.makedirs(labelsTr_folder, exist_ok=True)

    # Determine the path for the JSON file in the output folder
    json_file_path = os.path.join(output_folder, 'file_correspondence.json')

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            sub_id = subfolder  # Use the subfolder name directly as the sub_id
            rename_files_in_subfolder(subfolder_path, sub_id, imagesTr_folder, labelsTr_folder, json_file_path, segs, mods, seg_id, change_ids)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rename files in subfolders and save the correspondence to a JSON file.')
    parser.add_argument('--input_folder', type=str, help='Path to the main folder containing subfolders with files to rename', required=True)
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to save the renamed files and JSON file')
    parser.add_argument('--segs', action='store_true', help='Flag indicating that the files to rename are label maps and should not have the _0000 suffix')
    parser.add_argument('--mod', nargs='+', help='List of substrings to identify each modality')
    parser.add_argument('--seg_id', nargs='+', help='List of substrings to identify segmentations/label maps')
    parser.add_argument('--change_ids', action='store_true', help='Flag indicating whether to change subject IDs')
    args = parser.parse_args()

    process_main_folder(args.input_folder, args.output_folder, args.segs, args.mod, args.seg_id, args.change_ids)