import os
import shutil
import argparse

def reorganize_files(input_dir, output_dir, create_subdirs, file_substring, file_extension):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(file_extension) and file_substring in filename:
            if create_subdirs:
                # Extract the subject ID from the filename
                subject_id = filename.split('_')[0]
                
                # Create a subfolder for the subject in the output directory
                subject_dir = os.path.join(output_dir, subject_id)
                os.makedirs(subject_dir, exist_ok=True)
                
                # Copy the file to the subject's subfolder
                shutil.copy(os.path.join(input_dir, filename), os.path.join(subject_dir, filename))
                print("Copied %s to %s" % (filename, os.path.join(subject_dir, filename)))
                
            else:
                # Copy the file to the output directory without creating subfolders
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
                print("Copied %s to %s" % (filename, os.path.join(output_dir, filename)))

    print("Files have been successfully copied and organized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize all the files present in the input directory into either a single output directory or an output directory containing subdirectories based on subject ID.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the input directory containing files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory.")
    parser.add_argument("-s", "--create_subdirs", action="store_true", help="Flag to create subdirectories for each subject ID.")
    parser.add_argument("-f", "--file", default="", help="Substring to filter files to move.")
    parser.add_argument("-e", "--extension", default=".nii.gz", help="File extension to filter files to move. Default is .nii.gz")

    args = parser.parse_args()

    reorganize_files(args.input_dir, args.output_dir, args.create_subdirs, args.file, args.extension)