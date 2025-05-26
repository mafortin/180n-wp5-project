import nibabel as nib
import numpy as np
from PIL import Image

def create_mri_gif_pil(nifti_file, output_gif, axis=2, duration=100):
    # Load NIfTI file
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Normalize data to 0–255
    data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data.astype(np.uint8)

    # Extract slices along chosen axis and convert to RGB frames
    frames = []
    for i in range(data.shape[axis]):
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:
            slice_2d = data[:, :, i]

        rgb = np.stack([slice_2d]*3, axis=-1)  # Convert to RGB
        frame = Image.fromarray(rgb)
        frames.append(frame)

    # Save animated GIF with proper duration (duration in ms per frame)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # ms per frame (e.g., 100 = 10 FPS)
        loop=0
    )

# Example usage
create_mri_gif_pil(
    '/home/marcantf/Data/scaifield-bids-resub/derivatives/ntnu/SFP_TH_099/ses-01/segm/N4_coreg_mprage/mri/orig.mgz',
    '/home/marcantf/Figures/mri_cor_40ms.gif',
    duration=40  # X ms/frame (e.g., 200 ms per frame = 5 FPS)
)
