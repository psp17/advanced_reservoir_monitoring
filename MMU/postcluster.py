import os
import pandas as pd
import shutil
from pathlib import Path

def extract_fid_quarter_year(entry):
    """
    Extract FID, quarter, and year from entry.
    Handles two formats:
    - '110_1_19' -> FID 110, Quarter 1, Year 2019
    - '116_all' -> FID 116, Quarter 'all', Year None (match all)
    """
    if pd.isna(entry) or entry == '':
        return None, None, None
    
    entry_str = str(entry).strip()
    parts = entry_str.split('_')
    
    # Handle "116_all" format
    if len(parts) == 2 and parts[1].lower() == 'all':
        try:
            fid = int(parts[0])
            return fid, 'all', None  # Return 'all' as quarter
        except ValueError:
            return None, None, None
    
    # If format is fid_quarter_year
    elif len(parts) == 3:
        try:
            fid = int(parts[0])
            quarter = int(parts[1])
            year_short = int(parts[2])
            
            # Convert 2-digit year to 4-digit year
            if year_short < 100:
                year = 2000 + year_short
            else:
                year = year_short
                
            return fid, quarter, year
        except ValueError:
            return None, None, None
    
    return None, None, None

def find_matching_files(input_dir, fid, quarter=None, year=None):
    """
    Find all files matching FID, quarter, and year pattern.
    If quarter is 'all', match all files with that FID.
    Matches files *starting with* the pattern.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"  ERROR: Input directory does not exist: {input_dir}")
        return []
    
    # If quarter is 'all', match all files with that FID
    if quarter == 'all':
        pattern = f"FID{fid}_"
        search_pattern = f"{pattern}*.tif"
        print(f"  Searching for pattern: {search_pattern}")
        matching_files = list(input_path.glob(search_pattern)) # CHANGED
        
        if not matching_files:
            # CHANGED: Fixed typo from .tif to *.tif to find all tif files
            all_files = list(input_path.glob("*.tif")) 
            print(f"  No matches found. Total .tif files in directory: {len(all_files)}")
            if len(all_files) > 0 and len(all_files) <= 5:
                print(f"  Sample files: {[f.name for f in all_files[:5]]}")
        
        return matching_files
    
    # Otherwise, match specific FID_Quarter_Year
    pattern = f"FID{fid}_Q{quarter}_{year}"
    search_pattern = f"{pattern}*.tif"
    print(f"  Searching for pattern: {search_pattern}")
    matching_files = list(input_path.glob(search_pattern)) 
    
    if not matching_files:
        # Try alternate patterns
        alt_patterns = [
            f"FID{fid}_q{quarter}_{year}",  # lowercase q
            f"{fid}_Q{quarter}_{year}",     # no FID prefix
            f"{fid}_q{quarter}_{year}",     # no FID, lowercase q
        ]
        
        for alt_pattern in alt_patterns:
            search_pattern = f"{alt_pattern}*.tif"
            alt_matches = list(input_path.glob(search_pattern)) 
            if alt_matches:
                print(f"  Found matches with alternate pattern: {search_pattern}")
                matching_files = alt_matches
                break
    
    return matching_files

def find_cluster_outputs(output_dirs, base_filename):
    """Find cluster output files for a given base filename."""
    cluster_files = {}
    base_name = base_filename.stem
    
    for method, output_dir in output_dirs.items():
        output_path = Path(output_dir)
        # Look for cleaned cluster outputs
        cluster_file = output_path / f"{base_name}_{method}_cleaned.tif"
        if cluster_file.exists():
            cluster_files[method] = cluster_file
    
    return cluster_files

def delete_files(file_list):
    """Delete a list of files."""
    deleted_count = 0
    for file_path in file_list:
        try:
            if file_path.exists():
                os.remove(file_path)
                print(f"  Deleted: {file_path.name}")
                deleted_count += 1
        except Exception as e:
            print(f"  Error deleting {file_path.name}: {e}")
    return deleted_count

def move_files(file_list, destination_dir):
    """Move a list of files to destination directory."""
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for file_path in file_list:
        try:
            if file_path.exists():
                dest_file = dest_path / file_path.name
                shutil.move(str(file_path), str(dest_file))
                print(f"  Moved: {file_path.name} -> {destination_dir}")
                moved_count += 1
        except Exception as e:
            print(f"  Error moving {file_path.name}: {e}")
    return moved_count

def copy_files(file_list, destination_dir):
    """Copy a list of files to destination directory."""
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for file_path in file_list:
        try:
            if file_path.exists():
                dest_file = dest_path / file_path.name
                shutil.copy2(str(file_path), str(dest_file))
                print(f"  Copied: {file_path.name} -> {destination_dir}")
                copied_count += 1
        except Exception as e:
            print(f"  Error copying {file_path.name}: {e}")
    return copied_count

def process_excel_file(excel_path, input_dir, output_dirs, base_output_dir):
    """Process Excel file and organize files based on categories."""
    
    print(f"Reading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    print(f"\nColumns found: {df.columns.tolist()}")
    
    # Statistics
    stats = {
        'Noisy': {'moved_input': 0, 'deleted_clusters': 0},
        'Good Cluster': {'moved_input': 0, 'moved_clusters': 0},
        'K means good': {'moved_input': 0, 'moved_clusters': 0},
        'Report': {'copied_input': 0},
        'Moisture': {'copied_input': 0}
    }
    
    # Process each category
    for column in df.columns:
        category = column.strip()
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")
        
        entries = df[column].dropna()
        print(f"Found {len(entries)} entries")
        
        for entry in entries:
            fid, quarter, year = extract_fid_quarter_year(entry)
            
            if fid is None:
                continue
            
            # Display what we're processing
            if quarter is None and year is None:
                print(f"\nProcessing: All files with FID{fid}")
            else:
                print(f"\nProcessing: FID{fid}_Q{quarter}_{year}")
            
            # Find matching input files
            input_files = find_matching_files(input_dir, fid, quarter, year)
            
            if not input_files:
                print(f"  Warning: No input files found for {entry}")
                continue
            
            print(f"  Found {len(input_files)} input file(s)")
            
            # Process based on category
            if category.lower() == 'noisy':
                # Move input files to noisy folder
                noisy_dir = Path(base_output_dir) / 'noisy'
                moved = move_files(input_files, noisy_dir)
                stats['Noisy']['moved_input'] += moved
                
                # # Delete cluster outputs
                # for input_file in input_files:
                #     cluster_files = find_cluster_outputs(output_dirs, input_file)
                #     deleted_clusters = delete_files(cluster_files.values())
                #     stats['Noisy']['deleted_clusters'] += deleted_clusters
            
            elif category.lower() == 'good cluster':
                # Move input files
                recluster_dir = Path(base_output_dir) / 'good cluster' / 'input'
                moved = move_files(input_files, recluster_dir)
                stats['Good Cluster']['moved_input'] += moved
                
                # Move cluster outputs
                for input_file in input_files:
                    cluster_files = find_cluster_outputs(output_dirs, input_file)
                    for method, cluster_file in cluster_files.items():
                        method_dir = Path(base_output_dir) / 'good cluster' / method
                        moved_clusters = move_files([cluster_file], method_dir)
                        stats['Good Cluster']['moved_clusters'] += moved_clusters
            
            elif category.lower() == 'kmeans good' or category.lower() == 'k means good':
                # Move input files to 'border' folder
                border_dir = Path(base_output_dir) / 'border' / 'input'
                moved = move_files(input_files, border_dir)
                stats['K means good']['moved_input'] += moved
                
                # Move cluster outputs
                for input_file in input_files:
                    cluster_files = find_cluster_outputs(output_dirs, input_file)
                    for method, cluster_file in cluster_files.items():
                        method_dir = Path(base_output_dir) / 'border' / method
                        moved_clusters = move_files([cluster_file], method_dir)
                        stats['K means good']['moved_clusters'] += moved_clusters
            
            elif category.lower() == 'moisture':
                # Copy input files only
                moisture_dir = Path(base_output_dir) / 'moisture'
                copied = copy_files(input_files, moisture_dir)
                stats['Moisture']['copied_input'] += copied
            
            elif category.lower() == 'report':
                # Copy input files only
                report_dir = Path(base_output_dir) / 'report'
                copied = copy_files(input_files, report_dir)
                stats['Report']['copied_input'] += copied
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nNoisy:")
    print(f"  Input files moved: {stats['Noisy']['moved_input']}")
    print(f"  Cluster files deleted: {stats['Noisy']['deleted_clusters']}")
    
    print(f"\nRecluster:")
    print(f"  Input files moved: {stats['Good Cluster']['moved_input']}")
    print(f"  Cluster files moved: {stats['Good Cluster']['moved_clusters']}")
    
    print(f"\nK means good (Border):")
    print(f"  Input files moved: {stats['K means good']['moved_input']}")
    print(f"  Cluster files moved: {stats['K means good']['moved_clusters']}")
    
    print(f"\nMoisture:")
    print(f"  Input files copied: {stats['Moisture']['copied_input']}")
    
    print(f"\nReport:")
    print(f"  Input files copied: {stats['Report']['copied_input']}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")

def main():
    """Main execution function."""
    
    # Configuration - UPDATE THESE PATHS
    excel_path = "/home/arm/Downloads/Unimodal.xlsx"
    input_dir = "/home/arm/Documents/ARM/output/Unimodal"
    
    output_dirs = {
        'kmeans': "/data/OUTPUT/Clustering/OrganizedUnimodal/ClusterOutput/kmeans/cleaned_clusters1",
        'gmm': "/data/OUTPUT/Clustering/OrganizedUnimodal/ClusterOutput/gmm/cleaned_clusters1",
        'fuzzy': "/data/OUTPUT/Clustering/OrganizedUnimodal/ClusterOutput/fuzzy/cleaned_clusters1",
        'ensemble': "/data/OUTPUT/Clustering/OrganizedUnimodal/ClusterOutput/ensemble/cleaned_clusters"
    }
    
    base_output_dir = "/data/OUTPUT/Clustering/UnimodalGood"
    
    # Verify paths exist
    if not Path(excel_path).exists():
        print(f"Error: Excel file not found at {excel_path}")
        print("Please update the excel_path variable with the correct path.")
        return
    
    if not Path(input_dir).exists():
        print(f"Error: Input directory not found at {input_dir}")
        return
    
    # Process the Excel file
    process_excel_file(excel_path, input_dir, output_dirs, base_output_dir)

if __name__ == "__main__":
    main()

