"""
Python executable script to extract old subject IDs corresponding to new subject IDs and fetch specific files from dataset subfolders.

This script is designed to:
1. **Extract old subject IDs** corresponding to new subject IDs from a CSV file based on filenames.
2. **Filter subject IDs** by a dataset identifier (e.g., the first letter or a custom prefix of the new subject ID).
3. **Fetch specific files** related to the matched old subject IDs from dataset subfolders, with options to filter files based on a substring in the filename.
4. **Handle special cases** such as datasets from the Human Connectome Project (HCP), where the data is located inside a subfolder (e.g., 'T1w').

Flags:
- `-c, --csv`        : Path to the CSV file containing 'new_subject_id' and 'old_subject_id' mappings.
- `-f, --folder`     : Path to the folder containing `.nii.gz` files used for extracting the new subject IDs.
- `--did`            : Filter files by the initial letter or prefix of the new subject ID (e.g., "A" for "A005").
- `-d, --dataset-folder`: Path to the main folder containing subfolders named after `old_subject_id`, where data is stored.
- `-s, --substring`  : Substring to search for in filenames of the dataset files (e.g., 'T1w', 'brain', etc.).
- `--hcp`            : Flag to specify if the dataset is from the HCP, which uses subfolders like 'T1w' to organize the data.

Example Usage:
python script.py -c subjects.csv -f /path/to/nii/files --did A -d /path/to/dataset/folder -s "T1w" --hcp

Authors: MAF and ChatGPT, November 2024
"""

import os
import shutil
import pandas as pd
import argparse

# Main function to handle arguments
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Extract old subject IDs corresponding to new subject IDs and fetch specific files from dataset subfolders.')

    # Argument for the path to the CSV file
    parser.add_argument('-c', '--csv', type=str, required=True, help='Path to the CSV file containing subject information')

    # Argument for the folder containing the .nii.gz files
    parser.add_argument('-f', '--folder', type=str, required=True, help='Path to the folder containing .nii.gz files')

    # Argument for filtering by dataset letter (initial letter of the new subject ID)
    parser.add_argument('--did', type=str, help='Filter files by the initial letter or prefix of the new subject ID (e.g., "A" for "A005")')

    # Argument for dataset folder path where files are stored
    parser.add_argument('-d', '--dataset-folder', type=str, required=True, help='Path to the folder containing subfolders for each old_subject_id')

    # Argument for the substring in the filenames to search for
    parser.add_argument('-s', '--substring', type=str, required=True, help='Substring to search for in filenames (e.g., "T1w")')

    # Flag for specifying if the dataset is from HCP
    parser.add_argument('--hcp', action='store_true', help='Specify if the dataset is from HCP, where images are inside the "T1w" subfolder')

    # New flag for copying matched files to a specified directory
    parser.add_argument('--copy-to', type=str, help='Directory where the matched files will be copied to (if present)')

    # Parse the arguments
    args = parser.parse_args()

    # Extract matched old subject IDs
    new_subject_ids, old_subject_ids = extract_old_subject_ids(args.csv, args.folder, args.did)

    # Print the matched subject IDs
    print("Matched new_subject_ids:", new_subject_ids)
    print("Matched old_subject_ids:", old_subject_ids)

    # Fetch and optionally copy the files for the matched subjects
    fetch_files(args.dataset_folder, new_subject_ids, old_subject_ids, args.substring, args.hcp, args.copy_to)


# Function to extract old subject IDs based on new subject IDs
def extract_old_subject_ids(csv_file, folder_path, dataset_filter=None):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Get the list of filenames in the directory
    files_in_directory = os.listdir(folder_path)

    # Initialize lists to store matched new_subject_ids and old_subject_ids
    matched_new_subject_ids = []
    matched_old_subject_ids = []

    # Iterate over the files in the directory
    for file_name in files_in_directory:
        if file_name.endswith('.nii.gz'):
            # Extract the part between 'aseg_' and '.nii.gz', e.g., 'aseg_A005.nii.gz' -> 'A005'
            new_subject_id = file_name.split('aseg_')[1].split('.nii.gz')[0]

            # Check if we should filter by a specific dataset (if provided)
            if dataset_filter and not new_subject_id.startswith(dataset_filter):
                continue  # Skip this file if it doesn't match the filter

            # Check if the new_subject_id exists in the 'new_subject_id' column
            if new_subject_id in df['new_subject_id'].values:
                # Find the corresponding old_subject_id
                old_subject_id = df.loc[df['new_subject_id'] == new_subject_id, 'old_subject_id'].values[0]
                matched_new_subject_ids.append(new_subject_id)  # Store new_subject_id
                matched_old_subject_ids.append(old_subject_id)  # Store corresponding old_subject_id

    return matched_new_subject_ids, matched_old_subject_ids


# Function to fetch files based on old_subject_ids, substring, and HCP flag
def fetch_files(dataset_folder, new_subject_ids, old_subject_ids, substring, is_hcp, copy_to=None):
    # Check if --copy-to flag is provided
    if copy_to:
        # Create /prep/ directory inside the provided directory
        prep_directory = os.path.join(copy_to, 'prep')
        os.makedirs(prep_directory, exist_ok=True)

    # Loop through each new_subject_id and old_subject_id
    for new_subject_id, old_subject_id in zip(new_subject_ids, old_subject_ids):
        # Use old_subject_id to construct the path for the subject folder in the dataset folder
        subject_folder = os.path.join(dataset_folder, old_subject_id)  # Using old_subject_id for folder search

        # If HCP dataset, check for the 'T1w' subfolder
        if is_hcp:
            subject_folder = os.path.join(subject_folder, 'T1w')

        # Check if the subject folder exists
        if not os.path.exists(subject_folder):
            print(f"Folder for {old_subject_id} does not exist: {subject_folder}")
            continue

        # Get a list of files in the subject folder
        files_in_subject_folder = os.listdir(subject_folder)

        # Filter files based on the substring in their filename
        matching_files = [f for f in files_in_subject_folder if substring in f]

        # Copy the matched files if the --copy-to flag is provided
        for file_name in matching_files:
            source_file = os.path.join(subject_folder, file_name)
            print(f"Matched file: {source_file}")

            if copy_to:
                # Create a subfolder for the new_subject_id inside /prep/
                subject_copy_folder = os.path.join(prep_directory, new_subject_id)
                os.makedirs(subject_copy_folder, exist_ok=True)

                # Copy the file to the new subject folder
                destination_file = os.path.join(subject_copy_folder, file_name)
                shutil.copy2(source_file, destination_file)
                print(f"Copied {file_name} to {destination_file}")


# Run the script
if __name__ == "__main__":
    main()
