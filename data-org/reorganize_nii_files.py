import os
import shutil
import argparse
import json
import re  # for regex matching

def reorganize_files(input_dir, output_dir, create_subdirs, file_substring, file_extension, copy_only_json, sub_id_pattern, input_has_subdirs):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load subject IDs from the JSON file if provided
    subject_ids = set()
    if copy_only_json:
        with open(copy_only_json, 'r') as json_file:
            data = json.load(json_file)
            raw_subject_ids = data['subject_id']
            subject_ids = set(raw_subject_ids)  # keep as-is, includes image_ prefix if present

    # Get the list of files to process
    if input_has_subdirs:
        files_to_process = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                files_to_process.append(os.path.join(root, file))
    else:
        files_to_process = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]

    # Iterate over all files
    for file_path in files_to_process:
        filename = os.path.basename(file_path)

        if filename.endswith(file_extension) and file_substring in filename:
            subject_id = None

            # --- Pattern-based subject ID extraction ---
            if sub_id_pattern == "prefix":
                subject_id = filename.split('_')[0]

            elif sub_id_pattern == "between_image_and_niigz":
                if "image_" in filename and ".nii.gz" in filename:
                    subject_id = "image_" + filename.split('image_')[1].split('.nii.gz')[0].split('_')[0]
                else:
                    print(f"Skipping file {filename} (no match for 'between_image_and_niigz').")
                    continue

            elif sub_id_pattern == "between_image_and_xxxx":
                if "image_" in filename and "_xxxx" in filename:
                    subject_id = "image_" + filename.split('image_')[1].split('_xxxx')[0]
                else:
                    print(f"Skipping file {filename} (no match for 'between_image_and_xxxx').")
                    continue

            elif sub_id_pattern == "image_yyy":
                # Match the entire 'image_...' part before the 4-digit block
                match = re.search(r"(image_.+?)_\d{4}", filename)
                if match:
                    subject_id = match.group(1)
                else:
                    print(f"Skipping file {filename} (no match for 'image_yyy').")
                    continue

            else:
                raise ValueError("Unsupported subject ID pattern specified.")

            # Skip files if subject ID is not in the JSON list
            if copy_only_json and subject_id not in subject_ids:
                continue

            # --- Copy logic ---
            if create_subdirs:
                subject_dir = os.path.join(output_dir, subject_id)
                os.makedirs(subject_dir, exist_ok=True)
                shutil.copy(file_path, os.path.join(subject_dir, filename))
                print(f"Copied {filename} → {subject_dir}")
            else:
                shutil.copy(file_path, os.path.join(output_dir, filename))
                print(f"Copied {filename} → {output_dir}")

    print("✅ Files have been successfully copied and organized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize files into a structured output directory based on subject IDs."
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the input directory containing files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory.")
    parser.add_argument("-s", "--create_subdirs", action="store_true", help="Flag to create subdirectories for each subject ID.")
    parser.add_argument("-f", "--file", default="", help="Substring to filter files to move.")
    parser.add_argument("-e", "--extension", default=".nii.gz", help="File extension to filter files to move. Default is .nii.gz")
    parser.add_argument("-j", "--copy_only_json", help="Path to a JSON file containing a list of subject IDs to filter files to move.")
    parser.add_argument(
        "-p", "--sub_id_pattern",
        default="prefix",
        choices=["prefix", "between_image_and_niigz", "between_image_and_xxxx", "image_yyy"],
        help="Pattern to extract the subject ID from the filename."
    )
    parser.add_argument("--input_has_subdirs", action="store_true", help="Flag to indicate if the input directory contains subdirectories.")

    args = parser.parse_args()

    reorganize_files(
        args.input_dir,
        args.output_dir,
        args.create_subdirs,
        args.file,
        args.extension,
        args.copy_only_json,
        args.sub_id_pattern,
        args.input_has_subdirs
    )
