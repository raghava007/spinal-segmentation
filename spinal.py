#!/usr/bin/env python3
"""Save sagittal view with C1-C7 vertebrae labeled."""

import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import center_of_mass


def process_cervical_spine(ct_path: str, seg_dir: str, output_dir: str) -> str:
    """
    Process CT scan and generate labeled sagittal view of C1-C7 vertebrae.
    
    Args:
        ct_path: Path to CT NIfTI file (.nii or .nii.gz)
        seg_dir: Directory containing vertebrae segmentation files
        output_dir: Directory to save output image
        
    Returns:
        Path to saved output image
    """
    # Load CT
    img = nib.load(ct_path)
    data = img.get_fdata()

    print(f"CT Shape: {data.shape}")

    # Load all C1-C7 segmentations
    vertebrae = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    combined_mask = np.zeros_like(data)
    centroids = {}

    for i, v in enumerate(vertebrae, 1):
        filepath = os.path.join(seg_dir, f'vertebrae_{v}.nii.gz')
        if os.path.exists(filepath):
            seg = nib.load(filepath).get_fdata()
            combined_mask[seg > 0] = i
            if np.sum(seg > 0) > 0:
                com = center_of_mass(seg > 0)
                centroids[v] = com
                print(f"{v}: center at ({com[0]:.0f}, {com[1]:.0f}, {com[2]:.0f})")

    if not centroids:
        raise ValueError("No vertebrae segmentations found in seg_dir")

    # Find the best sagittal slice (through spine center)
    mid_x = int(np.mean([c[0] for c in centroids.values()]))
    print(f"\nUsing sagittal slice at x={mid_x}")

    # Create figure with labeled sagittal view
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Sagittal slice through spine
    sagittal_ct = data[mid_x, :, :].T
    sagittal_mask = combined_mask[mid_x, :, :].T.copy().astype(float)
    sagittal_mask[sagittal_mask == 0] = np.nan

    # Left: CT only
    axes[0].imshow(sagittal_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    axes[0].set_title('Original CT - Sagittal View', fontsize=14)
    axes[0].axis('off')

    # Right: CT with segmentation overlay and labels
    axes[1].imshow(sagittal_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    axes[1].imshow(sagittal_mask, cmap='tab10', alpha=0.5, origin='lower')

    # Add labels at centroid positions
    for i, v in enumerate(vertebrae):
        if v in centroids:
            y, z = centroids[v][1], centroids[v][2]
            axes[1].annotate(v, (y, z), fontsize=12, fontweight='bold',
                            color='yellow', ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    axes[1].set_title('C1-C7 Cervical Vertebrae Identified', fontsize=14)
    axes[1].axis('off')

    ct_filename = os.path.basename(ct_path)
    plt.suptitle(f'Cervical Spine Segmentation - {ct_filename}', fontsize=16)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sagittal_c1_c7_labeled.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    return output_path


if __name__ == '__main__':
    # Hardcoded paths
    ct_path = '/Users/fc20024/Documents/github/spinal-segmentation/sample input files/sub-gl003_dir-ax_ct.nii.gz'
    seg_dir = '/tmp/test_seg_gl003'
    output_dir = '/Users/fc20024/Documents/github/spinal-segmentation/output'
    
    process_cervical_spine(ct_path, seg_dir, output_dir)
