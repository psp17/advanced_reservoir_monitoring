import os
import shutil
from pathlib import Path

# --- Utility Functions ---

def parse_quarter_year(qy_string):
    """
    Parse quarter.year string like '1.19' or '2.19'
    Returns (quarter, year) tuple
    """
    parts = qy_string.strip().split('.')
    if len(parts) == 2:
        quarter = int(parts[0])
        year_short = int(parts[1])
        # Convert 2-digit year to 4-digit
        year = 2000 + year_short if year_short < 100 else year_short
        return (quarter, year)
    return None

def expand_range(start_q, start_y, end_q, end_y):
    """
    Expand a range of quarters into a list of (quarter, year) tuples.
    Example: Q1 2015 to Q3 2017 -> [(1,2015), (2,2015), (3,2015), (4,2015), (1,2016), ...]
    """
    quarters = []
    
    current_q = start_q
    current_y = start_y
    
    while (current_y < end_y) or (current_y == end_y and current_q <= end_q):
        quarters.append((current_q, current_y))
        
        # Move to next quarter
        current_q += 1
        if current_q > 4:
            current_q = 1
            current_y += 1
    
    return quarters

def parse_text_file_entries(text_file_path):
    """
    Parse text file with formats:
    - FID (move all files for this FID)
    - FID: Q.YY, Q.YY (individual quarters)
    - FID: till Q.YY (from Q1 2015 to Q.YY)
    - FID: before Q.YY (from Q1 2015 to quarter before Q.YY)
    - FID: Q.YY to Q.YY (range)
    - FID: Q.YY-YY (same quarter across year range, e.g., 2.19-25 = Q2 for 2019-2025)
    - FID: ex Q.YY, Q.YY (exclude specified quarters, move all others)
    
    Returns dict: {fid: {'quarters': [(q, y), ...], 'exclude': [(q, y), ...]} or 'ALL'}
    """
    fid_entries = {}
    
    try:
        with open(text_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if it's just a FID number (no colon)
                if ':' not in line:
                    try:
                        fid = int(line)
                        fid_entries[fid] = 'ALL'
                        print(f"  ✓ FID {fid}: ALL files")
                        continue
                    except ValueError:
                        print(f"  ⚠️  Line {line_num}: Invalid FID '{line}'")
                        continue
                
                # Split by colon to get FID and rest
                parts = line.split(':', 1)
                try:
                    fid = int(parts[0].strip())
                except ValueError:
                    print(f"  ⚠️  Line {line_num}: Invalid FID '{parts[0].strip()}'")
                    continue
                
                rest = parts[1].strip()
                
                if fid not in fid_entries:
                    fid_entries[fid] = {'quarters': [], 'exclude': []}
                
                # Handle "ex" (exclude) format - move all EXCEPT specified quarters
                if rest.lower().startswith('ex'):
                    exclude_part = rest.lower().replace('ex', '').strip()
                    
                    # Parse excluded quarters
                    quarter_parts = exclude_part.split(',')
                    excluded = []
                    for qp in quarter_parts:
                        qp = qp.strip()
                        if qp:
                            qy = parse_quarter_year(qp)
                            if qy:
                                excluded.append(qy)
                    
                    if excluded:
                        fid_entries[fid] = {'quarters': 'ALL_EXCEPT', 'exclude': excluded}
                        print(f"  ✓ FID {fid}: ALL files EXCEPT {len(excluded)} quarters")
                    continue
                
                # Handle "before Q.YY" format (from Q1 2015 till the quarter before specified)
                if rest.lower().startswith('before'):
                    before_part = rest.lower().replace('before', '').strip()
                    end_qy = parse_quarter_year(before_part)
                    if end_qy:
                        # Calculate till the quarter before (same as "till Q3 2021" if input is "before 4.21")
                        till_q = end_qy[0] - 1
                        till_y = end_qy[1]
                        if till_q < 1:
                            till_q = 4
                            till_y -= 1
                        
                        # Start from Q1 2015
                        quarters = expand_range(1, 2015, till_q, till_y)
                        fid_entries[fid]['quarters'].extend(quarters)
                        print(f"  ✓ FID {fid}: Q1 2015 till Q{till_q} {till_y} ({len(quarters)} quarters)")
                    else:
                        print(f"  ⚠️  Line {line_num}: Invalid 'before' format '{rest}'")
                    continue
                
                # Handle "till Q.YY" format (from Q1 2015 to specified quarter)
                if rest.lower().startswith('till'):
                    end_part = rest.lower().replace('till', '').strip()
                    end_qy = parse_quarter_year(end_part)
                    if end_qy:
                        # Start from Q1 2015
                        quarters = expand_range(1, 2015, end_qy[0], end_qy[1])
                        fid_entries[fid]['quarters'].extend(quarters)
                        print(f"  ✓ FID {fid}: Q1 2015 till Q{end_qy[0]} {end_qy[1]} ({len(quarters)} quarters)")
                    else:
                        print(f"  ⚠️  Line {line_num}: Invalid 'till' format '{rest}'")
                    continue
                
                # Handle "Q.YY-YY" format (same quarter across year range)
                # Example: 2.19-25 means Q2 for years 2019, 2020, 2021, 2022, 2023, 2024, 2025
                if '-' in rest and ' to ' not in rest.lower():
                    year_range_parts = rest.split('-')
                    if len(year_range_parts) == 2:
                        # Parse start: Q.YY
                        start_qy = parse_quarter_year(year_range_parts[0])
                        if start_qy:
                            quarter = start_qy[0]
                            start_year = start_qy[1]
                            
                            # Parse end year (just YY)
                            try:
                                end_year_short = int(year_range_parts[1].strip())
                                end_year = 2000 + end_year_short if end_year_short < 100 else end_year_short
                                
                                # Add the same quarter for each year in the range
                                for year in range(start_year, end_year + 1):
                                    fid_entries[fid]['quarters'].append((quarter, year))
                                
                                num_years = end_year - start_year + 1
                                print(f"  ✓ FID {fid}: Q{quarter} for years {start_year}-{end_year} ({num_years} quarters)")
                            except ValueError:
                                print(f"  ⚠️  Line {line_num}: Invalid year range format '{rest}'")
                        else:
                            print(f"  ⚠️  Line {line_num}: Invalid quarter-year range format '{rest}'")
                        continue
                
                # Handle "Q.YY to Q.YY" range format
                if ' to ' in rest.lower():
                    range_parts = rest.lower().split(' to ')
                    if len(range_parts) == 2:
                        start_qy = parse_quarter_year(range_parts[0])
                        end_qy = parse_quarter_year(range_parts[1])
                        
                        if start_qy and end_qy:
                            quarters = expand_range(start_qy[0], start_qy[1], 
                                                   end_qy[0], end_qy[1])
                            fid_entries[fid]['quarters'].extend(quarters)
                            print(f"  ✓ FID {fid}: Q{start_qy[0]} {start_qy[1]} to Q{end_qy[0]} {end_qy[1]} ({len(quarters)} quarters)")
                        else:
                            print(f"  ⚠️  Line {line_num}: Invalid range format '{rest}'")
                    continue
                
                # Handle comma-separated individual quarters
                quarter_parts = rest.split(',')
                for qp in quarter_parts:
                    qp = qp.strip()
                    if qp:
                        qy = parse_quarter_year(qp)
                        if qy:
                            fid_entries[fid]['quarters'].append(qy)
                        else:
                            print(f"  ⚠️  Line {line_num}: Invalid quarter format '{qp}'")
    
    except FileNotFoundError:
        print(f"ERROR: Text file not found at {text_file_path}")
        return {}
    
    return fid_entries

def find_matching_input_files(input_dir, fid, quarter=None, year=None):
    """
    Find input files matching FID and optionally quarter/year.
    Filename format: Indices_FID{fid}_Y{year}_Q{quarter}.tif
    
    Returns list of matching file paths.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        return []
    
    matching_files = []
    
    # If no quarter/year specified, get ALL files for this FID
    if quarter is None or year is None:
        pattern = f"Indices_FID{fid}_*.tif"
        matches = list(input_path.glob(pattern))
        matching_files.extend(matches)
        return matching_files
    
    # Match specific quarter and year
    pattern = f"Indices_FID{fid}_Y{year}_Q{quarter}.tif"
    matches = list(input_path.glob(pattern))
    matching_files.extend(matches)
    
    return matching_files

def find_matching_label_files(label_dir, fid, quarter=None, year=None):
    """
    Find label files matching FID and optionally quarter/year.
    Filename format: Indices_FID{fid}_Y{year}_Q{quarter}_ensemble_mmu.tif
    
    Returns list of matching file paths.
    """
    label_path = Path(label_dir)
    if not label_path.exists():
        return []
    
    matching_files = []
    
    # If no quarter/year specified, get ALL files for this FID
    if quarter is None or year is None:
        pattern = f"Indices_FID{fid}_*.tif"
        matches = list(label_path.glob(pattern))
        matching_files.extend(matches)
        return matching_files
    
    # Match specific quarter and year
    pattern = f"Indices_FID{fid}_Y{year}_Q{quarter}_ensemble_mmu.tif"
    matches = list(label_path.glob(pattern))
    matching_files.extend(matches)
    
    return matching_files

def get_all_quarters_for_fid(input_dir, label_dir, fid):
    """
    Get all available quarters for a FID by scanning both input and label directories.
    Returns set of (quarter, year) tuples.
    """
    quarters = set()
    
    # Scan input directory
    input_path = Path(input_dir)
    if input_path.exists():
        for file in input_path.glob(f"Indices_FID{fid}_*.tif"):
            # Parse: Indices_FID{fid}_Y{year}_Q{quarter}.tif
            name = file.stem
            parts = name.split('_')
            if len(parts) >= 4:
                try:
                    year_part = parts[2]  # Y2018
                    quarter_part = parts[3]  # Q4
                    year = int(year_part[1:])
                    quarter = int(quarter_part[1:])
                    quarters.add((quarter, year))
                except (ValueError, IndexError):
                    pass
    
    # Scan label directory
    label_path = Path(label_dir)
    if label_path.exists():
        for file in label_path.glob(f"Indices_FID{fid}_*.tif"):
            name = file.stem
            parts = name.split('_')
            if len(parts) >= 4:
                try:
                    year_part = parts[2]  # Y2018
                    quarter_part = parts[3]  # Q4
                    year = int(year_part[1:])
                    quarter = int(quarter_part[1:])
                    quarters.add((quarter, year))
                except (ValueError, IndexError):
                    pass
    
    return quarters

def move_files(file_list, destination_dir):
    """
    Move a list of files to destination directory.
    Returns count of successfully moved files.
    """
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for file_path in file_list:
        if file_path and file_path.exists():
            try:
                dest_file = dest_path / file_path.name
                shutil.move(str(file_path), str(dest_file))
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving {file_path.name}: {e}")
    return moved_count

def process_text_file(text_file_path, source_input_dir, source_label_dir, 
                     dest_input_dir, dest_label_dir):
    """Process single text file and move matching input and label files."""
    
    text_path = Path(text_file_path)
    if not text_path.exists():
        print(f"ERROR: Text file not found: {text_file_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing: {text_path.name}")
    print(f"{'='*60}\n")
    
    # Statistics
    stats = {
        'total_fids': 0,
        'moved_inputs': 0,
        'moved_labels': 0,
        'missing_entries': 0
    }
    
    # Parse entries from text file
    print("Parsing entries...")
    fid_entries = parse_text_file_entries(text_path)
    
    if not fid_entries:
        print(f"No valid entries found in {text_path.name}")
        return
    
    stats['total_fids'] = len(fid_entries)
    print(f"\nTotal FIDs to process: {stats['total_fids']}\n")
    print("Moving files...\n")
    
    # Process each FID
    for fid, entry in fid_entries.items():
        if entry == 'ALL':
            # Move ALL files for this FID
            input_files = find_matching_input_files(source_input_dir, fid)
            label_files = find_matching_label_files(source_label_dir, fid)
            
            if not input_files and not label_files:
                print(f"❌ MISSING: FID {fid} (no files found)")
                stats['missing_entries'] += 1
                continue
            
            moved_inputs = move_files(input_files, dest_input_dir)
            moved_labels = move_files(label_files, dest_label_dir)
            
            if moved_inputs > 0 or moved_labels > 0:
                print(f"✓ Moved FID {fid} (ALL): {moved_inputs} input(s), {moved_labels} label(s)")
            stats['moved_inputs'] += moved_inputs
            stats['moved_labels'] += moved_labels
            
        elif entry['quarters'] == 'ALL_EXCEPT':
            # Move all files EXCEPT excluded quarters
            all_quarters = get_all_quarters_for_fid(source_input_dir, source_label_dir, fid)
            excluded_quarters = set(entry['exclude'])
            quarters_to_move = all_quarters - excluded_quarters
            
            if not quarters_to_move:
                print(f"❌ MISSING: FID {fid} (no files to move after exclusions)")
                stats['missing_entries'] += 1
                continue
            
            total_inputs = 0
            total_labels = 0
            
            for quarter, year in sorted(quarters_to_move):
                input_files = find_matching_input_files(source_input_dir, fid, quarter, year)
                label_files = find_matching_label_files(source_label_dir, fid, quarter, year)
                
                moved_inputs = move_files(input_files, dest_input_dir)
                moved_labels = move_files(label_files, dest_label_dir)
                
                total_inputs += moved_inputs
                total_labels += moved_labels
            
            print(f"✓ Moved FID {fid} (excluding {len(excluded_quarters)} quarters): {total_inputs} input(s), {total_labels} label(s)")
            stats['moved_inputs'] += total_inputs
            stats['moved_labels'] += total_labels
            
        else:
            # Move specific quarters
            quarters = entry['quarters']
            for quarter, year in quarters:
                input_files = find_matching_input_files(source_input_dir, fid, quarter, year)
                label_files = find_matching_label_files(source_label_dir, fid, quarter, year)
                
                if not input_files and not label_files:
                    print(f"  ❌ MISSING: FID {fid}, Q{quarter}, {year}")
                    stats['missing_entries'] += 1
                    continue
                
                moved_inputs = move_files(input_files, dest_input_dir)
                moved_labels = move_files(label_files, dest_label_dir)
                
                if moved_inputs > 0 or moved_labels > 0:
                    print(f"  ✓ Moved FID {fid}, Q{quarter}, {year}: {moved_inputs} input(s), {moved_labels} label(s)")
                stats['moved_inputs'] += moved_inputs
                stats['moved_labels'] += moved_labels
    
    # Print summary
    print(f"\n{'='*60}")
    print("✨ SUMMARY ✨")
    print(f"{'='*60}")
    print(f"Total FIDs processed: {stats['total_fids']}")
    print(f"Input files moved: {stats['moved_inputs']}")
    print(f"Label files moved: {stats['moved_labels']}")
    print(f"Missing entries: {stats['missing_entries']}")
    print(f"{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")

def main():
    """Main execution function."""
    
    # Configuration - UPDATE THESE PATHS
    text_file_path = "/home/arm/Documents/ARM/text.txt"
    source_input_dir = "/home/arm/Documents/ARM/Validation_Sen2indices"
    source_label_dir = "/home/arm/Documents/ARM/Validation_Sen2indices/ClusterOutput/ensemble/mmu_filtered"
    dest_input_dir = "/data/backup/val_indices_noisy_s2/input"
    dest_label_dir = "/data/backup/val_indices_noisy_s2/label"
    
    # Path Verification
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Text file: {text_file_path}")
    print(f"Source Input: {source_input_dir}")
    print(f"Source Labels: {source_label_dir}")
    print(f"Destination Input: {dest_input_dir}")
    print(f"Destination Labels: {dest_label_dir}")
    print(f"{'='*60}")
    
    # Verify paths exist
    for path_name, path_str in [
        ("Text file", text_file_path),
        ("Source input directory", source_input_dir),
        ("Source label directory", source_label_dir)
    ]:
        if not Path(path_str).exists():
            print(f"\n🛑 Error: {path_name} not found: {path_str}")
            return
    
    # Process the text file
    process_text_file(
        text_file_path,
        source_input_dir,
        source_label_dir,
        dest_input_dir,
        dest_label_dir
    )

if __name__ == "__main__":
    main()