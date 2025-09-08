import nibabel as nib
import numpy as np
from PIL import Image

def create_mri_gif_pil(
    nifti_file,
    output_gif,
    axis=2,
    duration=100,
    vmin=None,
    vmax=None
):
    # Load NIfTI file
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # If vmin/vmax not provided, use full range
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # Clip to window and normalize to 0–255
    data = np.clip(data, vmin, vmax)
    data = 255 * (data - vmin) / (vmax - vmin)
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

        rgb = np.stack([slice_2d] * 3, axis=-1)  # grayscale → RGB
        frame = Image.fromarray(rgb)
        frames.append(frame)

    # Save animated GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


create_mri_gif_pil(
    '/home/marcantf/Data/scaifield-bids-resub/derivatives/ntnu/SFP_TH_099/ses-01/segm/N4_coreg_mprage/mri/orig.mgz',
    '/home/marcantf/Figures/mri_ax_40ms_new_window.gif',
    duration=40,  # X ms/frame (e.g., 200 ms per frame = 5 FPS)
    axis=1,
    vmin=15,
    vmax=150
)
