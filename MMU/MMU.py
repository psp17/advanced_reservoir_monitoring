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
ensemble_folder = "/home/arm/Documents/ARM/AfterMMUScreen/afterMMU_PreprocessRequired/recluster/final_label"
output_folder = '/home/arm/Documents/ARM/AfterMMUScreen/afterMMU_PreprocessRequired/recluster/final_label_clean'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

def extract_fid_qq_yyyy(filename):
    """Extract FID_QQ_YYYY from filename"""
    # Pattern: FID_QQ_YYYY
    match = re.search(r'(\d+_\d+_\d{4})', filename)
    if match:
        return match.group(1)
    return "Unknown"

def find_optimal_threshold_2bin_gap(region_sizes, num_features):
    """
    Find optimal threshold using 2-bin gap method.
    Sets threshold at the START of the first valid 2-bin gap.
    Also returns max component size for additional filtering.
    """
    max_size = region_sizes.max()
    mean_size = region_sizes.mean()
    median_size = np.median(region_sizes)
    
    # Histogram Bin Gap Analysis
    num_bins = min(50, max(20, num_features // 5))
    hist, bin_edges = np.histogram(region_sizes, bins=num_bins)
    
    # Find 2-bin gaps (exactly 2 consecutive zero bins)
    two_bin_gaps = []
    
    for i in range(len(hist) - 1):
        # Check if we have exactly 2 consecutive zero bins
        if i + 1 < len(hist) and hist[i] == 0 and hist[i + 1] == 0:
            # Check that this is the start of a 2-bin gap (not part of a longer gap)
            is_start = (i == 0 or hist[i - 1] > 0)
            is_end = (i + 2 >= len(hist) or hist[i + 2] > 0)
            
            if is_start and is_end:
                gap_start_bin = i
                gap_end_bin = i + 1
                start_value = bin_edges[gap_start_bin]
                end_value = bin_edges[gap_end_bin + 1]
                center = (start_value + end_value) / 2
                
                # Validate the gap
                has_data_before = gap_start_bin > 0 and hist[gap_start_bin - 1] > 0
                has_data_after = gap_end_bin < len(hist) - 1 and hist[gap_end_bin + 1] > 0
                in_lower_range = center < max_size * 0.8
                
                if has_data_before and has_data_after and in_lower_range:
                    two_bin_gaps.append({
                        'start_bin': gap_start_bin,
                        'end_bin': gap_end_bin,
                        'start_value': start_value,
                        'end_value': end_value,
                        'center': center
                    })
    
    # Use the first valid 2-bin gap (leftmost, closest to small components)
    if len(two_bin_gaps) > 0:
        # Sort by center position (use the earliest gap)
        two_bin_gaps.sort(key=lambda x: x['center'])
        best_gap = two_bin_gaps[0]
        gap_threshold = best_gap['start_value']
        method = f"2-Bin Gap (at {gap_threshold:.0f}px)"
    else:
        # Fallback: use P20 if no 2-bin gap found
        gap_threshold = np.percentile(region_sizes, 20)
        method = "Fallback P20 (no 2-bin gap)"
    
    # Safety checks
    p10 = np.percentile(region_sizes, 10)
    if gap_threshold < p10:
        gap_threshold = p10
        method = f"{method}→P10"
    
    return gap_threshold, method, max_size

def process_single_file(file_info):
    """Process a single ensemble file with 2-bin gap + 10% max size filter + min 2 components"""
    filepath = file_info['path']
    filename = file_info['filename']
    fid_qq_yyyy = extract_fid_qq_yyyy(filename)
    
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
                    'fid_qq_yyyy': fid_qq_yyyy,
                    'status': 'skipped',
                    'reason': 'no components',
                    'original': 0,
                    'kept': 0,
                    'removed': 0
                }
            
            # NEW: Check if ORIGINAL image has less than 2 components
            # If so, copy as-is without any filtering
            if num_features < 2:
                output_path = os.path.join(output_folder, filename)
                shutil.copy2(filepath, output_path)
                
                return {
                    'filename': filename,
                    'fid_qq_yyyy': fid_qq_yyyy,
                    'status': 'copied_as_is',
                    'reason': f'original image has only {num_features} component(s)',
                    'original': num_features,
                    'kept': num_features,
                    'removed': 0
                }
            
            # Calculate region sizes
            region_sizes = np.bincount(labeled_array.ravel())[1:]
            
            # Find optimal threshold using 2-bin gap method
            gap_threshold, method, max_component_size = find_optimal_threshold_2bin_gap(
                region_sizes, num_features
            )
            
            # Calculate 10% of max component size
            size_threshold = max_component_size * 0.10
            
            # Use the more restrictive threshold (higher value)
            final_threshold = max(gap_threshold, size_threshold)
            
            # Determine which threshold was used
            if size_threshold > gap_threshold:
                threshold_used = "10% Max Size"
                threshold_type = "10% Max"
            else:
                threshold_used = method
                threshold_type = "2-Bin Gap"
            
            # Filter small regions
            filtered_labels = labeled_array.copy()
            removed_count = 0
            for region_id in range(1, num_features + 1):
                if region_sizes[region_id - 1] < final_threshold:
                    filtered_labels[filtered_labels == region_id] = 0
                    removed_count += 1
            
            # Count kept components
            kept_count = num_features - removed_count
            
            # NEW LOGIC: If after filtering we have less than 2 components,
            # keep the largest 2 components from the original image
            fallback_to_top2 = False
            if kept_count < 2:
                fallback_to_top2 = True
                # Get all component sizes with their IDs
                component_sizes = [(i+1, region_sizes[i]) for i in range(len(region_sizes))]
                # Sort by size (descending)
                component_sizes.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top 2 (or top 1 if only 1 component exists)
                top_n = min(2, len(component_sizes))
                top_labels = [x[0] for x in component_sizes[:top_n]]
                
                # Create new filtered labels with only top 2
                filtered_labels = labeled_array.copy()
                for region_id in range(1, num_features + 1):
                    if region_id not in top_labels:
                        filtered_labels[filtered_labels == region_id] = 0
                
                kept_count = top_n
                removed_count = num_features - kept_count
            
            # If more than 5 components, keep only the 5 largest
            additional_removed = 0
            if kept_count > 5:
                # Get sizes of remaining components
                remaining_labels = np.unique(filtered_labels)[1:]  # Exclude 0
                remaining_sizes = []
                for label_id in remaining_labels:
                    size = np.sum(filtered_labels == label_id)
                    remaining_sizes.append((label_id, size))
                
                # Sort by size (descending) and keep top 5
                remaining_sizes.sort(key=lambda x: x[1], reverse=True)
                top_5_labels = [x[0] for x in remaining_sizes[:5]]
                
                # Remove all except top 5
                for label_id in remaining_labels:
                    if label_id not in top_5_labels:
                        filtered_labels[filtered_labels == label_id] = 0
                        additional_removed += 1
                
                kept_count = 5
            
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
                'fid_qq_yyyy': fid_qq_yyyy,
                'status': 'success',
                'original': num_features,
                'kept': kept_count,
                'removed': removed_count,
                'additional_removed': additional_removed,
                'threshold': final_threshold,
                'threshold_type': threshold_type,
                'method': threshold_used,
                'max_size': max_component_size,
                'gap_threshold': gap_threshold,
                'size_threshold': size_threshold,
                'fallback_to_top2': fallback_to_top2
            }
            
    except Exception as e:
        return {
            'filename': filename,
            'fid_qq_yyyy': fid_qq_yyyy,
            'status': 'error',
            'error': str(e)
        }

# Main execution
print("="*80)
print("2-BIN GAP + 10% MAX SIZE IMAGE CLEANING (TOP-2 FALLBACK)")
print("="*80)
print(f"Input folder:  {ensemble_folder}")
print(f"Output folder: {output_folder}")
print("="*80)

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
    
    # Step 3: Organize results by FID_QQ_YYYY
    fid_results = defaultdict(list)
    for result in results:
        fid_qq_yyyy = result.get('fid_qq_yyyy', 'Unknown')
        fid_results[fid_qq_yyyy].append(result)
    
    # Step 4: Print summary
    print("\n" + "="*80)
    print("[3/3] PROCESSING SUMMARY")
    print("="*80)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    copied_as_is_count = 0
    copied_as_is_files = []
    fallback_count = 0
    fallback_files = []
    total_original = 0
    total_kept = 0
    total_removed = 0
    total_additional_removed = 0
    gap_used_count = 0
    size_used_count = 0
    
    # Print FID_QQ_YYYY wise threshold usage
    print("\n" + "="*80)
    print("FID_QQ_YYYY WISE THRESHOLD USAGE")
    print("="*80)
    
    for fid_qq_yyyy in sorted(fid_results.keys()):
        fid_results_list = fid_results[fid_qq_yyyy]
        
        gap_count = sum(1 for r in fid_results_list if r['status'] == 'success' and r.get('threshold_type') == '2-Bin Gap')
        size_count = sum(1 for r in fid_results_list if r['status'] == 'success' and r.get('threshold_type') == '10% Max')
        success_in_fid = sum(1 for r in fid_results_list if r['status'] == 'success')
        skipped_in_fid = sum(1 for r in fid_results_list if r['status'] == 'skipped')
        copied_in_fid = sum(1 for r in fid_results_list if r['status'] == 'copied_as_is')
        fallback_in_fid = sum(1 for r in fid_results_list if r['status'] == 'success' and r.get('fallback_to_top2', False))
        error_in_fid = sum(1 for r in fid_results_list if r['status'] == 'error')
        
        print(f"\n{fid_qq_yyyy}:")
        print(f"  Total files: {len(fid_results_list)}")
        print(f"  Successfully processed: {success_in_fid}")
        print(f"  Copied as-is (<2 comp): {copied_in_fid}")
        print(f"  Fallback to top-2: {fallback_in_fid}")
        print(f"  Skipped: {skipped_in_fid}")
        print(f"  Errors: {error_in_fid}")
        if success_in_fid > 0:
            print(f"  Threshold Usage:")
            print(f"    - 2-Bin Gap: {gap_count} files ({100*gap_count/success_in_fid:.1f}%)")
            print(f"    - 10% Max Size: {size_count} files ({100*size_count/success_in_fid:.1f}%)")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS BY FILE")
    print("="*80)
    
    for result in results:
        if result['status'] == 'success':
            success_count += 1
            total_original += result['original']
            total_kept += result['kept']
            total_removed += result['removed']
            total_additional_removed += result.get('additional_removed', 0)
            
            # Count which threshold was more restrictive
            if result.get('threshold_type') == '10% Max':
                size_used_count += 1
            else:
                gap_used_count += 1
            
            # Check if fallback was used
            if result.get('fallback_to_top2', False):
                fallback_count += 1
                fallback_files.append(result['filename'])
                print(f"\n⚠ {result['filename']} (FALLBACK TO TOP-2)")
            else:
                print(f"\n✓ {result['filename']}")
            
            print(f"  FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Components: {result['original']} → {result['kept']} "
                  f"(removed {result['removed']})", end="")
            if result.get('additional_removed', 0) > 0:
                print(f" + {result['additional_removed']} extra (keeping top 5)")
            else:
                print()
            
            if not result.get('fallback_to_top2', False):
                print(f"  Max size: {result['max_size']:.0f}px | "
                      f"Gap T: {result['gap_threshold']:.0f}px | "
                      f"10% T: {result['size_threshold']:.0f}px")
                print(f"  Threshold Used: {result['threshold_type']}")
            else:
                print(f"  Note: Filtering left <2 components, kept largest 2 instead")
                
        elif result['status'] == 'copied_as_is':
            copied_as_is_count += 1
            copied_as_is_files.append(result['filename'])
            print(f"\n⊕ {result['filename']}")
            print(f"  FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Status: COPIED AS-IS ({result['reason']})")
            print(f"  Original components: {result['original']}")
        elif result['status'] == 'skipped':
            skipped_count += 1
            print(f"\n⊘ {result['filename']}")
            print(f"  FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Reason: {result['reason']}")
        else:
            error_count += 1
            print(f"\n✗ {result['filename']}")
            print(f"  FID_QQ_YYYY: {result['fid_qq_yyyy']}")
            print(f"  Error: {result.get('error', 'unknown')}")
    
    # Print list of copied-as-is files
    if copied_as_is_files:
        print("\n" + "="*80)
        print("FILES COPIED AS-IS (ORIGINAL HAD <2 COMPONENTS)")
        print("="*80)
        for i, filename in enumerate(copied_as_is_files, 1):
            print(f"{i:3d}. {filename}")
    
    # Print list of fallback files
    if fallback_files:
        print("\n" + "="*80)
        print("FILES WITH FALLBACK TO TOP-2 (FILTERING LEFT <2 COMPONENTS)")
        print("="*80)
        for i, filename in enumerate(fallback_files, 1):
            print(f"{i:3d}. {filename}")
    
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Total files:               {len(ensemble_files)}")
    print(f"Successfully cleaned:      {success_count}")
    print(f"  - Normal filtering:      {success_count - fallback_count}")
    print(f"  - Fallback to top-2:     {fallback_count}")
    print(f"Copied as-is (<2 comp):    {copied_as_is_count}")
    print(f"Skipped (no data):         {skipped_count}")
    print(f"Errors:                    {error_count}")
    print("-"*80)
    print(f"Threshold Usage:")
    print(f"  2-Bin Gap used:          {gap_used_count} files")
    print(f"  10% Max Size used:       {size_used_count} files")
    print("-"*80)
    if total_original > 0:
        print(f"Total components:          {total_original:,}")
        print(f"Components kept:           {total_kept:,} ({100*total_kept/total_original:.1f}%)")
        print(f"Components removed:        {total_removed:,} ({100*total_removed/total_original:.1f}%)")
        if total_additional_removed > 0:
            print(f"Additional removed (>5):   {total_additional_removed:,}")
    else:
        print(f"Total components:          0")
        print(f"Components kept:           0")
        print(f"Components removed:        0")
    print("="*80)
    print(f"\n✅ Cleaned images saved to: {output_folder}")
    print("="*80)