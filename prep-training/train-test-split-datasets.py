import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Main function to handle arguments
def main():
    parser = argparse.ArgumentParser(description='Split dataset into train and test sets.')

    # Argument for input base directory
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input directory containing subjects or files')
    # Optional argument for train output directory
    parser.add_argument('-tr', '--train', type=str, help="Path to the output train directory (default: 'train' subdir of input directory)")
    # Optional argument for test output directory
    parser.add_argument('-te', '--test', type=str, help="Path to the output test directory (default: 'test' subdir of input directory)")
    # Argument for test size
    parser.add_argument('-s', '--split', type=float, default=0.15, help='Test set size (default: 0.15 for 85% train, 15% test)')
    # Optional flag for no subdirectories (if subjects are files)
    parser.add_argument('--no-subdirs', action='store_true', help='Use this flag if the subjects are files and not in individual subdirectories')
    # Flag to save CSV files with subject IDs
    parser.add_argument('--save-csv', action='store_true', help='Save train and test subjects to CSV files')
    # Optional flag for CSV file path
    parser.add_argument('--csv-path', type=str, help="Directory to save the CSV files (default: same as input directory)")
    # Flag to save CSV files based on dataset name
    parser.add_argument('--dataset-named-csv', action='store_true', help='Use dataset-specific names for the CSV files')
    # Flag to skip file/directory moving or copying
    parser.add_argument('--skip-copy', action='store_true', help='Skip the file/directory moving or copying process')

    # Parse the arguments
    args = parser.parse_args()

    # Set default values for train, test directories, and CSV path if not provided
    train_dir = args.train if args.train else os.path.join(args.input, 'train')
    test_dir = args.test if args.test else os.path.join(args.input, 'test')
    csv_path = args.csv_path if args.csv_path else args.input

    # Call the function to split the dataset
    split_dataset(
        base_dir=args.input,
        train_dir=train_dir,
        test_dir=test_dir,
        test_size=args.split,
        no_subdirs=args.no_subdirs,
        save_csv=args.save_csv,
        csv_path=csv_path,
        dataset_named_csv=args.dataset_named_csv,
        skip_copy=args.skip_copy
    )

def split_dataset(base_dir, train_dir, test_dir, test_size=0.15, no_subdirs=False, save_csv=False, csv_path='', dataset_named_csv=False, skip_copy=False):
    # Create train and test directories if they don't exist
    if not skip_copy:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    if no_subdirs:
        # Case 2: All subject files are directly in the base_dir (no subdirectories)
        all_subjects = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
        print("Subjects detected in the dataset: ", all_subjects)

        def extract_subject_id(filename):
            """
            Extracts the subject ID from a filename.

            If the filename starts with 'aseg_' or 'image_' and ends with '.nii.gz',
            the subject ID is extracted as the substring after the prefix and before '.nii.gz'.

            If neither prefix is found, the subject ID is the filename minus '.nii.gz'.
            """
            if filename.endswith('.nii.gz'):
                return filename.rsplit('.nii.gz', 1)[0]  # Remove the extension

            return filename  # Fallback to the original filename if it doesn't match the pattern

    else:
        # Case 1: Each subject is in its own subdirectory
        all_subjects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subject_ids = all_subjects  # In this case, subject IDs are directory names
        print("Subjects detected in the dataset: ", subject_ids)

    # Handle cases where test_size is 0 or 1
    if test_size == 0:
        train_subjects = all_subjects
        test_subjects = []
    elif test_size == 1:
        train_subjects = []
        test_subjects = all_subjects
    else:
        # Split the subjects into train and test
        train_subjects, test_subjects = train_test_split(all_subjects, test_size=test_size, random_state=42)

    # Corresponding subject IDs for train/test
    train_subject_ids = [extract_subject_id(s) for s in train_subjects] if no_subdirs else train_subjects
    test_subject_ids = [extract_subject_id(s) for s in test_subjects] if no_subdirs else test_subjects

    # Move/copy subjects or files to their respective directories
    if not skip_copy:
        for subject in train_subjects:
            src = os.path.join(base_dir, subject)
            dst = os.path.join(train_dir, subject)
            if no_subdirs:
                shutil.copy(src, dst)
            else:
                shutil.copytree(src, dst)

        for subject in test_subjects:
            src = os.path.join(base_dir, subject)
            dst = os.path.join(test_dir, subject)
            if no_subdirs:
                shutil.copy(src, dst)
            else:
                shutil.copytree(src, dst)

        print(f"Training set: {len(train_subjects)} subjects")
        print(f"Test set: {len(test_subjects)} subjects")

    else:
        print(f"Skipping file/directory moving or copying. Train/Test splits computed only.")

    # Save train and test subject CSV files if the flag is set
    if save_csv:
        # Extract the last folder name from the input directory path (dataset name)
        dataset_name = os.path.basename(os.path.normpath(base_dir))

        # Paths for individual dataset CSVs
        train_csv_named = os.path.join(csv_path, f'{dataset_name}_train_subjects.csv')
        test_csv_named = os.path.join(csv_path, f'{dataset_name}_test_subjects.csv')

        # Paths for unified train/test CSVs
        train_csv_unified = os.path.join(csv_path, 'train_all_subjects.csv')
        test_csv_unified = os.path.join(csv_path, 'test_all_subjects.csv')

        # Modes for unified CSV files
        train_unified_mode = 'a' if os.path.exists(train_csv_unified) else 'w'
        test_unified_mode = 'a' if os.path.exists(test_csv_unified) else 'w'

        # Prepare the data for CSV
        train_data = pd.DataFrame({
            'subject_id': train_subject_ids,
            'dataset': [f'{dataset_name}'] * len(train_subject_ids)
        })

        test_data = pd.DataFrame({
            'subject_id': test_subject_ids,
            'dataset': [f'{dataset_name}'] * len(test_subject_ids)
        })

        # Save individual dataset-specific CSVs
        train_data.to_csv(train_csv_named, mode='w', header=True, index=False)
        test_data.to_csv(test_csv_named, mode='w', header=True, index=False)
        print(f"Dataset-specific Train subjects saved to: {train_csv_named}")
        print(f"Dataset-specific Test subjects saved to: {test_csv_named}")

        # Save unified train and test CSVs
        train_data.to_csv(train_csv_unified, mode=train_unified_mode, header=(train_unified_mode == 'w'), index=False)
        test_data.to_csv(test_csv_unified, mode=test_unified_mode, header=(test_unified_mode == 'w'), index=False)
        print(f"Unified Train subjects saved to: {train_csv_unified}")
        print(f"Unified Test subjects saved to: {test_csv_unified}")

if __name__ == "__main__":
    main()
