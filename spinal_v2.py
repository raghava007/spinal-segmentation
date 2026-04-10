#!/usr/bin/env python3
"""Save sagittal view with C1-C7 vertebrae labeled and axial view of selected vertebra."""

import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import center_of_mass


def process_cervical_spine(ct_path: str, target_vertebra: str = 'C1') -> str:
    """
    Process CT scan and identify target vertebra in sagittal and axial views.
    
    Args:
        ct_path: Path to CT NIfTI file (.nii or .nii.gz)
        target_vertebra: Which vertebra to identify (C1-C7)
        
    Returns:
        Path to saved output image
    """
    target_vertebra = target_vertebra.upper()
    
    # Derive seg_dir and output_dir from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seg_dir = os.path.join(script_dir, 'segm')
    output_dir = os.path.join(script_dir, 'output')
    
    # Load CT
    img = nib.load(ct_path)
    data = img.get_fdata()

    print(f"CT Shape: {data.shape}")
    print(f"Target vertebra: {target_vertebra}")

    # Load only the target vertebra segmentation
    filepath = os.path.join(seg_dir, f'vertebrae_{target_vertebra}.nii.gz')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Segmentation file not found: {filepath}")
    
    target_seg = nib.load(filepath).get_fdata()
    
    if np.sum(target_seg > 0) == 0:
        raise ValueError(f"No voxels found for {target_vertebra}")
    
    # Get centroid of target vertebra
    centroid = center_of_mass(target_seg > 0)
    target_x = int(centroid[0])
    target_y = int(centroid[1])
    target_z = int(centroid[2])
    print(f"{target_vertebra}: center at ({target_x}, {target_y}, {target_z})")

    # Create figure with sagittal and axial views
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Get sagittal slice at target vertebra centroid
    # Sagittal slice through target vertebra
    sagittal_ct = data[target_x, :, :].T
    
    # Get target vertebra mask for sagittal view
    sagittal_mask = target_seg[target_x, :, :].T.copy().astype(float)
    sagittal_mask[sagittal_mask == 0] = np.nan
    
    # Create colored overlay function
    def create_color_overlay(mask, color=(1, 0, 0), alpha=0.6):
        h, w = mask.shape
        overlay = np.zeros((h, w, 4))
        mask_bool = ~np.isnan(mask) & (mask > 0)
        overlay[mask_bool, 0] = color[0]
        overlay[mask_bool, 1] = color[1]
        overlay[mask_bool, 2] = color[2]
        overlay[mask_bool, 3] = alpha
        return overlay

    # TOP LEFT: Sagittal CT only
    axes[0, 0].imshow(sagittal_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    axes[0, 0].set_title(f'Original CT - Sagittal View (x={target_x})', fontsize=14)
    axes[0, 0].axis('off')

    # TOP RIGHT: Sagittal CT with target vertebra highlighted
    sagittal_overlay = create_color_overlay(sagittal_mask, color=(1, 0, 0), alpha=0.6)
    
    axes[0, 1].imshow(sagittal_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    axes[0, 1].imshow(sagittal_overlay, origin='lower')

    # Add label at centroid position for target vertebra
    axes[0, 1].annotate(target_vertebra, (target_y, target_z), fontsize=16, fontweight='bold',
                        color='yellow', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))

    axes[0, 1].set_title(f'{target_vertebra} Identified - Sagittal View', fontsize=14)
    axes[0, 1].axis('off')
    
    # BOTTOM: Axial view of target vertebra
    axial_ct = data[:, :, target_z].T
    
    # Get target vertebra mask for axial view
    axial_mask = target_seg[:, :, target_z].T.copy().astype(float)
    axial_mask[axial_mask == 0] = np.nan
    
    # BOTTOM LEFT: Axial CT only
    axes[1, 0].imshow(axial_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    axes[1, 0].set_title(f'Original CT - Axial View (z={target_z})', fontsize=14)
    axes[1, 0].axis('off')
    
    # BOTTOM RIGHT: Axial CT with target vertebra highlighted
    axial_overlay = create_color_overlay(target_seg[:, :, target_z].T.astype(float), color=(1, 0, 0), alpha=0.6)
    
    axes[1, 1].imshow(axial_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    axes[1, 1].imshow(axial_overlay, origin='lower')
    
    # Add label at centroid
    axes[1, 1].annotate(target_vertebra, (target_x, target_y), fontsize=16, fontweight='bold',
                        color='yellow', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    
    axes[1, 1].set_title(f'{target_vertebra} Identified - Axial View', fontsize=14)
    axes[1, 1].axis('off')

    ct_filename = os.path.basename(ct_path)
    plt.suptitle(f'Cervical Spine Segmentation - {ct_filename}', fontsize=16)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'cervical_spine_{target_vertebra.lower()}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    return output_path


if __name__ == '__main__':
    ct_path = '/Users/fc20024/Documents/github/spinal-segmentation/sample input files/sub-gl003_dir-ax_ct.nii.gz'
    target = 'C1'
    
    process_cervical_spine(ct_path, target)
    
