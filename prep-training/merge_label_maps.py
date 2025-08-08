import os
import argparse
import numpy as np
import nibabel as nib

def find_file_by_substring(directory, substring):
    """Find the first file in the directory that contains the given substring."""
    for fname in os.listdir(directory):
        if substring in fname:
            print(f"Found file for '{substring}': {fname}")
            return os.path.join(directory, fname)
    raise FileNotFoundError(f"No file containing '{substring}' found in {directory}")

def load_label_map(filepath):
    """Load a label map from .npy or .nii/.nii.gz file."""
    print(f"Loading label map: {filepath}")
    if filepath.endswith('.npy'):
        data = np.load(filepath)
        affine = None
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.int32)
        affine = img.affine
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    return data, affine

def save_label_map(data, affine, output_path):
    """Save the label map to .npy or .nii.gz format."""
    print(f"Saving merged label map to: {output_path}")
    if output_path.endswith('.npy'):
        np.save(output_path, data)
    elif output_path.endswith('.nii') or output_path.endswith('.nii.gz'):
        img = nib.Nifti1Image(data.astype(np.int32), affine)
        nib.save(img, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")

def merge_label_maps(ref_data, src_data, label_value):
    """Merge the specified label from src_data into ref_data where ref_data is zero."""
    print(f"Merging label {label_value} from source into reference map...")
    mask = (src_data == label_value) & (ref_data == 0)
    merged = ref_data.copy()
    merged[mask] = label_value
    print(f"Number of voxels added: {np.sum(mask)}")
    return merged

def main():
    parser = argparse.ArgumentParser(description="Merge a specific label from one 3D label map into another.")
    parser.add_argument('--input_dir', required=True, help='Directory containing the label maps')
    parser.add_argument('--file1', required=True, help='Substring or full name of the first label map file')
    parser.add_argument('--file2', required=True, help='Substring or full name of the second label map file')
    parser.add_argument('--reference', default=None, help='Filename of the reference label map (optional)')
    parser.add_argument('--label', type=int, required=True, help='Label value to merge from the second map')
    parser.add_argument('--output', required=True, help='Output filename (only the name, saved in input_dir)')

    args = parser.parse_args()

    path1 = find_file_by_substring(args.input_dir, args.file1)
    path2 = find_file_by_substring(args.input_dir, args.file2)

    # Determine reference and source
    if args.reference:
        ref_path = os.path.join(args.input_dir, args.reference)
        src_path = path2 if ref_path == path1 else path1
    else:
        ref_path, src_path = path1, path2

    print(f"Reference label map: {os.path.basename(ref_path)}")
    print(f"Source label map: {os.path.basename(src_path)}")

    ref_data, ref_affine = load_label_map(ref_path)
    src_data, _ = load_label_map(src_path)

    merged_data = merge_label_maps(ref_data, src_data, args.label)

    output_path = os.path.join(args.input_dir, args.output)
    save_label_map(merged_data, ref_affine, output_path)

    print("Merging complete.")

if __name__ == "__main__":
    main()
