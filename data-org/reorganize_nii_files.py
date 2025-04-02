import os
import shutil
import argparse
import json

def reorganize_files(input_dir, output_dir, create_subdirs, file_substring, file_extension, copy_only_json, sub_id_pattern, input_has_subdirs):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load subject IDs from the JSON file if provided
    subject_ids = set()

    if copy_only_json:
        with open(copy_only_json, 'r') as json_file:
            data = json.load(json_file)
            raw_subject_ids = data['subject_id']
            subject_ids = {s.replace('image_', '') for s in raw_subject_ids}

    # Get the list of files to process
    if input_has_subdirs:
        # Traverse subdirectories to find files
        files_to_process = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                files_to_process.append(os.path.join(root, file))
    else:
        # Process files directly in the input directory
        files_to_process = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Iterate over all files
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        
        if filename.endswith(file_extension) and file_substring in filename:
            # Extract the subject ID based on the provided pattern
            if sub_id_pattern == "prefix":
                subject_id = filename.split('_')[0]
            elif sub_id_pattern == "between_image_and_niigz":
                if "image_" in filename and ".nii.gz" in filename:
                    subject_id = filename.split('image_')[1].split('.nii.gz')[0].split('_')[0]
                else:
                    print(f"Skipping file {filename} as it does not match the 'between_image_and_niigz' pattern.")
                    continue
            elif sub_id_pattern == "between_image_and_xxxx":
                if "image_" in filename and "_xxxx" in filename:
                    subject_id = filename.split('image_')[1].split('_xxxx')[0]
                else:
                    print(f"Skipping file {filename} as it does not match the 'between_image_and_xxxx' pattern.")
                    continue
            else:
                raise ValueError("Unsupported subject ID pattern specified.")

            # Skip files if subject ID is not in the JSON file (when --copy_only_json is used)
            if copy_only_json and subject_id not in subject_ids:
                continue

            if create_subdirs:
                # Create a subfolder for the subject in the output directory
                subject_dir = os.path.join(output_dir, subject_id)
                os.makedirs(subject_dir, exist_ok=True)
                
                # Copy the file to the subject's subfolder
                shutil.copy(file_path, os.path.join(subject_dir, filename))
                print("Copied %s to %s" % (filename, os.path.join(subject_dir, filename)))
            else:
                # Copy the file to the output directory without creating subfolders
                shutil.copy(file_path, os.path.join(output_dir, filename))
                print("Copied %s to %s" % (filename, os.path.join(output_dir, filename)))

    print("Files have been successfully copied and organized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize all the files present in the input directory into either a single output directory or an output directory containing subdirectories based on subject ID.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the input directory containing files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory.")
    parser.add_argument("-s", "--create_subdirs", action="store_true", help="Flag to create subdirectories for each subject ID.")
    parser.add_argument("-f", "--file", default="", help="Substring to filter files to move.")
    parser.add_argument("-e", "--extension", default=".nii.gz", help="File extension to filter files to move. Default is .nii.gz")
    parser.add_argument("-j", "--copy_only_json", help="Path to a JSON file containing a list of subject IDs to filter files to move.")
    parser.add_argument("-p", "--sub_id_pattern", default="prefix", choices=["prefix", "between_image_and_niigz", "between_image_and_xxxx"], help="Pattern to extract the subject ID from the filename.")
    parser.add_argument("--input_has_subdirs", action="store_true", help="Flag to indicate if the input directory contains subdirectories.")

    args = parser.parse_args()

    reorganize_files(args.input_dir, args.output_dir, args.create_subdirs, args.file, args.extension, args.copy_only_json, args.sub_id_pattern, args.input_has_subdirs)
