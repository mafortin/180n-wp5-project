import os
import argparse

def replace_substring_in_filenames(directory, old_substring, new_substring):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if old_substring in file_name:
                new_file_name = file_name.replace(old_substring, new_substring)
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Replace a substring in filenames within a directory and its subfolders.')
    parser.add_argument('--dir', type=str, help='Path to the directory containing files to rename', required=True)
    parser.add_argument('--old_substring', type=str, help='Substring to be replaced', required=True)
    parser.add_argument('--new_substring', type=str, help='New substring to replace the old one', required=True)
    args = parser.parse_args()

    replace_substring_in_filenames(args.dir, args.old_substring, args.new_substring)