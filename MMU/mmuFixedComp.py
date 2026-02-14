import rasterio
import numpy as np
from scipy.ndimage import label
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import shutil
warnings.filterwarnings('ignore')

# Configuration - UPDATED PATHS
ensemble_folder = "/home/arm/Documents/ARM/AfterMMUScreen/afterMMU_PreprocessRequired/SLOPE"
output_folder = '/home/arm/Documents/ARM/AfterMMUScreen/afterMMU_PreprocessRequired/SLOPE/FINAL'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# ============================================================================
# COMPONENT SELECTION CONFIGURATION BY FID
# Format: FID : component_rank_to_keep (1=largest, 2=2nd largest, 3=3rd largest, etc.)
# OR: FID : number_of_top_components (for keeping multiple top components)
# Use negative numbers for specific ranks: -2 means "only 2nd largest"
# Use positive numbers for top N: 5 means "top 5 largest"
# ============================================================================
FID_COMPONENT_CONFIG = {
     '112': 1,      # Keep only largest
     '114': 1,      # Keep only largest
     '181': 6,     # Keep only largest
     '256': -3,     # Keep only largest 
    # '117': 1,     # Keep only largest
    # '181': 5,     # Keep top 5
    # '215': 5,     # Keep top 5
    # '267': -2,    # Keep ONLY 3rd largest component
    # '272': 1,     # Keep only largest
    # '32': -2,     # Keep ONLY 2nd largest component
    # '281': -2,    # Keep ONLY 3rd largest component
    # Add more as needed...
}

# Default number of components to keep if FID not in config
DEFAULT_COMPONENTS = 1

def extract_fid_from_filename(filename):
    """Extract FID from filename like FID181_Q2_2016"""
    match = re.search(r'FID(\d+)', filename)
    if match:
        return match.group(1)
    return None

def extract_fid_qq_yyyy(filename):
    """Extract FID_QQ_YYYY from filename like FID181_Q2_2016"""
    match = re.search(r'FID(\d+)_Q(\d+)_(\d{4})', filename)
    if match:
        fid = match.group(1)
        qq = match.group(2)
        yyyy = match.group(3)
        return f"{fid}_{qq}_{yyyy}"
    return "Unknown"

def get_specific_component(labeled_array, num_features, config_value):
    """
    Extract components based on configuration value:
    - Positive value (e.g., 5): Keep top N largest components
    - Negative value (e.g., -2): Keep ONLY the Nth largest component (2nd largest)
    
    Returns: filtered labeled array, count of kept components, count of removed components
    """
    if num_features == 0:
        return labeled_array, 0, 0
    
    # Get sizes of all components
    region_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (0)
    
    # Create list of (label_id, size) tuples
    component_sizes = [(i+1, region_sizes[i]) for i in range(len(region_sizes))]
    
    # Sort by size (descending)
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Determine which components to keep based on config_value
    if config_value > 0:
        # Positive: Keep top N components
        n_components = min(config_value, num_features)
        labels_to_keep = [x[0] for x in component_sizes[:n_components]]
    else:
        # Negative: Keep only the specific ranked component
        rank = abs(config_value)
        if rank > num_features:
            # Requested rank doesn't exist (e.g., asking for 3rd when only 2 exist)
            return labeled_array * 0, 0, num_features  # Return empty array
        labels_to_keep = [component_sizes[rank - 1][0]]  # rank-1 because 0-indexed
    
    # Create filtered array keeping only selected components
    filtered_labels = labeled_array.copy()
    for region_id in range(1, num_features + 1):
        if region_id not in labels_to_keep:
            filtered_labels[filtered_labels == region_id] = 0
    
    kept_count = len(labels_to_keep)
    removed_count = num_features - kept_count
    
    return filtered_labels, kept_count, removed_count

def process_single_file(file_info):
    """Process a single ensemble file, keeping specified components as per config"""
    filepath = file_info['path']
    filename = file_info['filename']
    fid_qq_yyyy = extract_fid_qq_yyyy(filename)
    fid = extract_fid_from_filename(filename)
    
    # Get the component configuration for this FID
    config_value = FID_COMPONENT_CONFIG.get(fid, DEFAULT_COMPONENTS)
    
    # Determine description of what we're keeping
    if config_value > 0:
        config_desc = f"top {config_value}"
    else:
        rank = abs(config_value)
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        config_desc = f"only {ordinal} largest"
    
    try:
        # Read the ensemble image with metadata
        with rasterio.open(filepath) as src:
            image = src.read(1)
            profile = src.profile.copy()
            
            # Label connected components
            structure = np.ones((3, 3), dtype=int)
            labeled_array, num_features = label(image == 1, structure=structure)
            
            if num_features == 0:
                return {
                    'filename': filename,
                    'fid': fid,
                    'fid_qq_yyyy': fid_qq_yyyy,
                    'status': 'skipped',
                    'reason': 'no components',
                    'original': 0,
                    'config_value': config_value,
                    'config_desc': config_desc,
                    'kept': 0,
                    'removed': 0
                }
            
            # Get specified component(s)
            filtered_labels, kept_count, removed_count = get_specific_component(
                labeled_array, num_features, config_value
            )
            
            # Check if requested component doesn't exist
            if kept_count == 0 and config_value < 0:
                rank = abs(config_value)
                return {
                    'filename': filename,
                    'fid': fid,
                    'fid_qq_yyyy': fid_qq_yyyy,
                    'status': 'skipped',
                    'reason': f'requested {rank}th component but only {num_features} exist',
                    'original': num_features,
                    'config_value': config_value,
                    'config_desc': config_desc,
                    'kept': 0,
                    'removed': num_features
                }
            
            # Create cleaned binary image
            cleaned_image = (filtered_labels > 0).astype(np.uint8)
            
            # Update profile for output
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=0
            )
            
            # Save cleaned image
            output_path = os.path.join(output_folder, filename)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(cleaned_image, 1)
            
            return {
                'filename': filename,
                'fid': fid,
                'fid_qq_yyyy': fid_qq_yyyy,
                'status': 'success',
                'original': num_features,
                'config_value': config_value,
                'config_desc': config_desc,
                'kept': kept_count,
                'removed': removed_count,
                'config_used': fid in FID_COMPONENT_CONFIG
            }
            
    except Exception as e:
        return {
            'filename': filename,
            'fid': fid,
            'fid_qq_yyyy': fid_qq_yyyy,
            'status': 'error',
            'config_value': config_value,
            'config_desc': config_desc,
            'error': str(e)
        }

# Main execution
print("="*80)
print("SPECIFIC COMPONENT EXTRACTION (CONFIGURED PER FID)")
print("="*80)
print(f"Input folder:  {ensemble_folder}")
print(f"Output folder: {output_folder}")
print(f"Default components to keep: {DEFAULT_COMPONENTS}")
print(f"Custom configurations: {len(FID_COMPONENT_CONFIG)} FID entries")
print("="*80)

# Print configuration
print("\nCONFIGURED COMPONENT SELECTION:")
print("-"*80)
for fid in sorted(FID_COMPONENT_CONFIG.keys(), key=lambda x: int(x)):
    config_val = FID_COMPONENT_CONFIG[fid]
    if config_val > 0:
        desc = f"Keep top {config_val} largest components"
    else:
        rank = abs(config_val)
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        desc = f"Keep ONLY {ordinal} largest component"
    print(f"  FID {fid}: {desc}")
print("-"*80)

# Step 1: Find all ensemble files
print("\n[1/3] Scanning ensemble files...")
all_ensemble_files = []
already_processed = []

for filename in os.listdir(ensemble_folder):
    if filename.endswith(".tif"):
        full_path = os.path.join(ensemble_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Check if already processed
        if os.path.exists(output_path):
            already_processed.append(filename)
        else:
            all_ensemble_files.append({
                'path': full_path,
                'filename': filename
            })

print(f"Found {len(all_ensemble_files) + len(already_processed)} total files")
print(f"Already processed: {len(already_processed)} files (skipping)")
print(f"To process: {len(all_ensemble_files)} files")

ensemble_files = all_ensemble_files

if len(ensemble_files) == 0:
    print("\n❌ No files to process!")
else:
    # Step 2: Process files in parallel
    num_workers = min(cpu_count(), len(ensemble_files))
    print(f"\n[2/3] Processing with {num_workers} workers...")
    print("-"*80)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, ensemble_files)
    
    # Step 3: Organize results by FID
    fid_results = defaultdict(list)
    for result in results:
        fid = result.get('fid', 'Unknown')
        fid_results[fid].append(result)
    
    # Step 4: Print summary
    print("\n" + "="*80)
    print("[3/3] PROCESSING SUMMARY")
    print("="*80)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    total_original = 0
    total_kept = 0
    total_removed = 0
    config_used_count = 0
    default_used_count = 0
    
    # Print FID wise summary
    print("\n" + "="*80)
    print("FID WISE SUMMARY")
    print("="*80)
    
    for fid in sorted(fid_results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        fid_results_list = fid_results[fid]
        
        success_in_fid = sum(1 for r in fid_results_list if r['status'] == 'success')
        skipped_in_fid = sum(1 for r in fid_results_list if r['status'] == 'skipped')
        error_in_fid = sum(1 for r in fid_results_list if r['status'] == 'error')
        
        # Get configuration for this FID
        config_value = FID_COMPONENT_CONFIG.get(fid, DEFAULT_COMPONENTS)
        if config_value > 0:
            config_desc = f"top {config_value}"
        else:
            rank = abs(config_value)
            ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            config_desc = f"only {ordinal}"
        
        config_status = "✓ CONFIGURED" if fid in FID_COMPONENT_CONFIG else "DEFAULT"
        
        print(f"\nFID {fid}: [{config_status} - Target: {config_desc}]")
        print(f"  Total files: {len(fid_results_list)}")
        print(f"  Successfully processed: {success_in_fid}")
        print(f"  Skipped: {skipped_in_fid}")
        print(f"  Errors: {error_in_fid}")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS BY FILE")
    print("="*80)
    
    for result in results:
        if result['status'] == 'success':
            success_count += 1
            total_original += result['original']
            total_kept += result['kept']
            total_removed += result['removed']
            
            if result.get('config_used', False):
                config_used_count += 1
                status_icon = "✓"
            else:
                default_used_count += 1
                status_icon = "○"
            
            print(f"\n{status_icon} {result['filename']}")
            print(f"  FID: {result['fid']} | FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Target: {result['config_desc']}")
            print(f"  Components: {result['original']} total → kept {result['kept']}, "
                  f"removed {result['removed']}")
                
        elif result['status'] == 'skipped':
            skipped_count += 1
            print(f"\n⊘ {result['filename']}")
            print(f"  FID: {result['fid']} | FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Reason: {result['reason']}")
        else:
            error_count += 1
            print(f"\n✗ {result['filename']}")
            print(f"  FID: {result['fid']} | FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Error: {result.get('error', 'unknown')}")
    
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Total files:               {len(ensemble_files)}")
    print(f"Successfully processed:    {success_count}")
    print(f"  - Using config:          {config_used_count}")
    print(f"  - Using default:         {default_used_count}")
    print(f"Skipped (no data):         {skipped_count}")
    print(f"Errors:                    {error_count}")
    print("-"*80)
    if total_original > 0:
        print(f"Total components:          {total_original:,}")
        print(f"Components kept:           {total_kept:,} ({100*total_kept/total_original:.1f}%)")
        print(f"Components removed:        {total_removed:,} ({100*total_removed/total_original:.1f}%)")
    else:
        print(f"Total components:          0")
        print(f"Components kept:           0")
        print(f"Components removed:        0")
    print("="*80)
    print(f"\n✅ Cleaned images saved to: {output_folder}")
    print("="*80)