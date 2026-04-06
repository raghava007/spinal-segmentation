#!/usr/bin/env python3
"""Simple NIfTI file viewer using matplotlib."""

import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import sys

def view_nii(filepath):
    """View a NIfTI file with interactive slice navigation."""
    img = nib.load(filepath)
    data = img.get_fdata()
    
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min: {data.min():.2f}, Max: {data.max():.2f}")
    
    # Create figure with sliders for each axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial slice indices (middle of each dimension)
    init_slices = [s // 2 for s in data.shape]
    
    # Display initial slices
    im_ax = axes[0].imshow(data[init_slices[0], :, :].T, cmap='gray', origin='lower')
    axes[0].set_title(f'Axial (dim 0): slice {init_slices[0]}')
    
    im_cor = axes[1].imshow(data[:, init_slices[1], :].T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (dim 1): slice {init_slices[1]}')
    
    im_sag = axes[2].imshow(data[:, :, init_slices[2]].T, cmap='gray', origin='lower')
    axes[2].set_title(f'Sagittal (dim 2): slice {init_slices[2]}')
    
    # Add sliders
    ax_slider0 = plt.axes([0.1, 0.15, 0.25, 0.03])
    ax_slider1 = plt.axes([0.4, 0.15, 0.25, 0.03])
    ax_slider2 = plt.axes([0.7, 0.15, 0.25, 0.03])
    
    slider0 = Slider(ax_slider0, 'Axial', 0, data.shape[0]-1, valinit=init_slices[0], valstep=1)
    slider1 = Slider(ax_slider1, 'Coronal', 0, data.shape[1]-1, valinit=init_slices[1], valstep=1)
    slider2 = Slider(ax_slider2, 'Sagittal', 0, data.shape[2]-1, valinit=init_slices[2], valstep=1)
    
    def update(val):
        im_ax.set_data(data[int(slider0.val), :, :].T)
        axes[0].set_title(f'Axial (dim 0): slice {int(slider0.val)}')
        
        im_cor.set_data(data[:, int(slider1.val), :].T)
        axes[1].set_title(f'Coronal (dim 1): slice {int(slider1.val)}')
        
        im_sag.set_data(data[:, :, int(slider2.val)].T)
        axes[2].set_title(f'Sagittal (dim 2): slice {int(slider2.val)}')
        
        fig.canvas.draw_idle()
    
    slider0.on_changed(update)
    slider1.on_changed(update)
    slider2.on_changed(update)
    
    plt.suptitle(filepath.split('/')[-1])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to sample file
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "sample input files/sub-gl003_dir-ax_ct.nii.gz")
    else:
        filepath = sys.argv[1]
    
    view_nii(filepath)
