import os
import argparse

def verify_directory(parent_dir, file_substring):
    # Get the list of subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    # Initialize a flag to track if all subdirectories contain the specified substring
    all_contain_substring = True
    
    # Iterate over each subdirectory
    for subdir in subdirectories:
        subdir_path = os.path.join(parent_dir, subdir)
        
        # Check if any file in the subdirectory contains the specified substring
        contains_substring = any(file_substring in filename for filename in os.listdir(subdir_path))
        
        if not contains_substring:
            print(f"Subdirectory '{subdir}' does not contain any files with the substring '{file_substring}'.")
            all_contain_substring = False
    
    if all_contain_substring:
        print(f"All subdirectories contain files with the substring '{file_substring}'.")
    else:
        print(f"Not all subdirectories contain files with the substring '{file_substring}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify if all subdirectories contain files with a certain substring.")
    parser.add_argument("-p", "--parent_dir", required=True, help="Path to the parent directory containing subdirectories.")
    parser.add_argument("-s", "--substring", required=True, help="Substring to look for in filenames.")

    args = parser.parse_args()

    verify_directory(args.parent_dir, args.substring)