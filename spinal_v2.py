#!/usr/bin/env python3
"""Save sagittal view with C1-C7 vertebrae labeled and axial view of selected vertebra."""

import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import shutil
from scipy.ndimage import center_of_mass


def clean_segmentation_folder(seg_dir: str):
    """Remove existing segmentation files before generating new ones."""
    if os.path.exists(seg_dir):
        print(f"Cleaning existing segmentation folder: {seg_dir}")
        shutil.rmtree(seg_dir)
    os.makedirs(seg_dir, exist_ok=True)


def run_totalsegmentator(ct_path: str, output_dir: str):
    """Run TotalSegmentator to generate vertebrae segmentation files."""
    print("Running TotalSegmentator to generate segmentation files...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run TotalSegmentator with vertebrae subset
    cmd = [
        'TotalSegmentator',
        '-i', ct_path,
        '-o', output_dir,
        '--roi_subset', 'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 
        'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("TotalSegmentator completed successfully.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TotalSegmentator failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("TotalSegmentator not found. Install with: pip install TotalSegmentator")


def process_cervical_spine(ct_path: str, target_vertebra: str = 'C1', sagittal_slice: int = None) -> str:
    """
    Process CT scan and identify target vertebra in sagittal and axial views.
    
    Args:
        ct_path: Path to CT NIfTI file (.nii or .nii.gz)
        target_vertebra: Which vertebra to identify (C1-C7)
        sagittal_slice: Optional specific sagittal slice index (x). If None, uses vertebra centroid.
        
    Returns:
        Path to saved output image
    """
    target_vertebra = target_vertebra.upper()
    
    # Derive seg_dir and output_dir from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seg_dir = os.path.join(script_dir, 'generate masked data')
    output_dir = os.path.join(script_dir, 'output')
    
    # Clean existing segmentation files before generating new ones
    clean_segmentation_folder(seg_dir)
    
    # Always run TotalSegmentator to generate fresh segmentation for this CT
    run_totalsegmentator(ct_path, seg_dir)
    
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
    centroid_x = int(centroid[0])
    target_y = int(centroid[1])
    target_z = int(centroid[2])
    
    # Use provided sagittal slice or fall back to centroid
    if sagittal_slice is not None:
        target_x = sagittal_slice
        print(f"Using provided sagittal slice: x={target_x}")
    else:
        target_x = centroid_x
    print(f"{target_vertebra}: center at ({centroid_x}, {target_y}, {target_z})")

    # Load all C1-C7 vertebrae for comprehensive view
    all_vertebrae = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    vertebrae_data = {}
    vertebrae_centroids = {}
    colors = {
        'C1': (1, 0, 0),      # Red
        'C2': (0, 1, 0),      # Green
        'C3': (0, 0, 1),      # Blue
        'C4': (1, 1, 0),      # Yellow
        'C5': (1, 0, 1),      # Magenta
        'C6': (0, 1, 1),      # Cyan
        'C7': (1, 0.5, 0),    # Orange
    }
    
    for vert in all_vertebrae:
        vert_path = os.path.join(seg_dir, f'vertebrae_{vert}.nii.gz')
        if os.path.exists(vert_path):
            vert_seg = nib.load(vert_path).get_fdata()
            if np.sum(vert_seg > 0) > 0:
                vertebrae_data[vert] = vert_seg
                vert_centroid = center_of_mass(vert_seg > 0)
                vertebrae_centroids[vert] = (int(vert_centroid[0]), int(vert_centroid[1]), int(vert_centroid[2]))
                print(f"{vert}: center at {vertebrae_centroids[vert]}")
    
    # Create figure with sagittal and axial views (3 rows x 2 cols)
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))

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

    # BOTTOM ROW: All C1-C7 vertebrae labeled
    # Use provided sagittal slice or average x of all centroids for all-vertebrae view
    if sagittal_slice is not None:
        avg_x = sagittal_slice
    elif vertebrae_centroids:
        avg_x = int(np.mean([c[0] for c in vertebrae_centroids.values()]))
    else:
        avg_x = target_x
    
    # LEFT: Sagittal view with all vertebrae
    sagittal_all_ct = data[avg_x, :, :].T
    axes[2, 0].imshow(sagittal_all_ct, cmap='gray', origin='lower', vmin=-400, vmax=1000)
    
    # Overlay all vertebrae with different colors
    for vert, vert_seg in vertebrae_data.items():
        vert_mask = vert_seg[avg_x, :, :].T.copy().astype(float)
        vert_mask[vert_mask == 0] = np.nan
        vert_overlay = create_color_overlay(vert_mask, color=colors[vert], alpha=0.5)
        axes[2, 0].imshow(vert_overlay, origin='lower')
        
        # Add label at centroid y, z position
        cy, cz = vertebrae_centroids[vert][1], vertebrae_centroids[vert][2]
        axes[2, 0].annotate(vert, (cy, cz), fontsize=12, fontweight='bold',
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[vert], alpha=0.8))
    
    axes[2, 0].set_title(f'All Cervical Vertebrae (C1-C7) - Sagittal View (x={avg_x})', fontsize=14)
    axes[2, 0].axis('off')
    
    # RIGHT: Legend and info
    axes[2, 1].axis('off')
    legend_y = 0.9
    axes[2, 1].text(0.5, 0.95, 'Vertebrae Legend', fontsize=16, fontweight='bold',
                    ha='center', va='top', transform=axes[2, 1].transAxes)
    
    for i, vert in enumerate(all_vertebrae):
        if vert in vertebrae_centroids:
            status = f"✓ Detected at {vertebrae_centroids[vert]}"
            color = colors[vert]
        else:
            status = "✗ Not found"
            color = (0.5, 0.5, 0.5)
        
        y_pos = 0.85 - (i * 0.1)
        # Draw color box
        rect = plt.Rectangle((0.1, y_pos - 0.03), 0.08, 0.06, 
                              facecolor=color, transform=axes[2, 1].transAxes, clip_on=False)
        axes[2, 1].add_patch(rect)
        axes[2, 1].text(0.22, y_pos, f'{vert}: {status}', fontsize=12,
                        ha='left', va='center', transform=axes[2, 1].transAxes)

    ct_filename = os.path.basename(ct_path)
    plt.suptitle(f'Cervical Spine Segmentation - {ct_filename}', fontsize=16)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    slice_suffix = f'_x{sagittal_slice}' if sagittal_slice is not None else ''
    output_path = os.path.join(output_dir, f'cervical_spine_{target_vertebra.lower()}{slice_suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    return output_path


if __name__ == '__main__':
    ct_path = '/Users/fc20024/Documents/github/spinal-segmentation/sample input files/sub-gl003_dir-ax_ct.nii.gz'
    target = 'C2'
    sagittal_slice = 200  # Set to a specific slice index (e.g., 250) or None to use centroid
    
    process_cervical_spine(ct_path, target, sagittal_slice)
    
