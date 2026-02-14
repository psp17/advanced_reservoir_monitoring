import sys
import numpy as np
from PIL import Image
from pathlib import Path
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- Configuration ---
# 1. Set your main directories
BASE_DIR = Path("/home/arm/Downloads/testinf")
OUTPUT_DIR = Path("/home/arm/Downloads/testinf/tiled_dataset")

# 2. Set your tile size
TILE_SIZE = 224

# 3. Set filtering threshold (0.0 means "keep all tiles")
#    (0.98 means "discard tiles that are > 98% one class")
FILTER_THRESHOLD = 0.95 
# --- End Configuration ---


def process_file_group(vv_path, base_dir, output_dir, tile_size, filter_threshold):
    """
    This is the "worker" function. It processes ONE file group.
    It's designed to be run in a parallel process.
    
    Returns: A string message ONLY if there is an error, else None.
    """
    try:
        # Define paths
        vv_dir = base_dir / "vv"
        vh_dir = base_dir / "vh"
        labels_dir = base_dir / "labels"
        out_vv = output_dir / "vv"
        out_vh = output_dir / "vh"
        out_labels = output_dir / "labels"

        filename = vv_path.name  # e.g., FID0...170925.png
        basename = vv_path.stem   # e.g., FID0...170925

        # --- Construct and find the matching filenames ---
        vh_path = vh_dir / filename
        label_matches = list(labels_dir.glob(f"{basename}*.png"))

        # --- Sanity Check 1: Do all files exist? ---
        if not vh_path.exists():
            return f"SKIPPED {basename}: Missing VH file. (Looked for: {vh_path})"
            
        if not label_matches:
            return f"SKIPPED {basename}: No label file found. (Looked for: {basename}*.png)"

        if len(label_matches) > 1:
            return f"SKIPPED {basename}: Found {len(label_matches)} possible labels. Please resolve ambiguity."
            
        label_path = label_matches[0]

        # --- Open, process, and save tiles ---
        with Image.open(vv_path) as img_vv, \
             Image.open(vh_path) as img_vh, \
             Image.open(label_path) as img_label:

            w, h = img_vv.size

            # --- Sanity Check 2: Are all images the same size? ---
            if (w, h) != img_vh.size or (w, h) != img_label.size:
                return f"SKIPPED {basename}: Mismatched sizes. VV: {img_vv.size}, VH: {img_vh.size}, Label: {img_label.size}"
            
            kept_tile_count = 0

            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    if (x + tile_size > w) or (y + tile_size > h):
                        continue
                    
                    box = (x, y, x + tile_size, y + tile_size)
                    tile_label = img_label.crop(box)

                    # --- Filtering ---
                    if filter_threshold > 0.0:
                        label_arr = np.array(tile_label)
                        unique, counts = np.unique(label_arr, return_counts=True)
                        if len(unique) == 1 or counts.max() / (tile_size * tile_size) > filter_threshold:
                            continue # Skip "boring" tile
                    # --- End Filtering ---

                    tile_vv = img_vv.crop(box)
                    tile_vh = img_vh.crop(box)
                    
                    out_filename = f"{basename}_tile_x{x}_y{y}.png"
                    
                    tile_vv.save(out_vv / out_filename)
                    tile_vh.save(out_vh / out_filename)
                    tile_label.save(out_labels / out_filename)
                    
                    kept_tile_count += 1
            
            # This indicates success
            return None 

    except Exception as e:
        # Catch any other unexpected errors during processing
        return f"ERROR processing {vv_path.name}: {e}"

def main():
    """
    Main function to set up and run the parallel processing.
    """
    # Define source and output dirs
    vv_dir = BASE_DIR / "vv"
    labels_dir = BASE_DIR / "labels"
    out_vv = OUTPUT_DIR / "vv"
    out_vh = OUTPUT_DIR / "vh"
    out_labels = OUTPUT_DIR / "labels"

    # Create output directories once, from the main process
    out_vv.mkdir(parents=True, exist_ok=True)
    out_vh.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"Starting parallel tile generation...")
    print(f"  Source: {BASE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  CPUs available: {multiprocessing.cpu_count()}")

    # Get the full list of "jobs"
    source_files = list(vv_dir.glob("*.png"))
    if not source_files:
        print(f"Error: No .png files found in {vv_dir}")
        print("Please check your BASE_DIR path.")
        sys.exit(1)
        
    print(f"Found {len(source_files)} file groups to process.")

    # "Partially" fill the worker function with the constant arguments
    # The only argument left "open" is 'vv_path'
    worker_func = partial(process_file_group, 
                          base_dir=BASE_DIR, 
                          output_dir=OUTPUT_DIR, 
                          tile_size=TILE_SIZE, 
                          filter_threshold=FILTER_THRESHOLD)

    # List to store any error messages from the workers
    errors = []

    # Create a process pool that uses all available CPUs
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        
        # 'imap' processes the list in order and is memory-efficient
        # 'tqdm' wraps this to create a live progress bar
        # 'desc' is the label for the progress bar
        print("--- Starting Work ---")
        job_iterator = pool.imap(worker_func, source_files)
        
        for result_message in tqdm(job_iterator, total=len(source_files), desc="Tiling images"):
            if result_message:
                errors.append(result_message)
    
    print("--- Work Complete ---")

    if errors:
        print(f"\n✅ Tiling complete, but with {len(errors)} warnings/errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✅ Tiling complete! All files processed successfully.")


if __name__ == "__main__":
    # This __name__ == "__main__" check is ESSENTIAL for multiprocessing
    # to work correctly on Windows.
    
    # Make sure required libraries are installed
    try:
        import numpy
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        print("Error: Required libraries not found.")
        print("Please run: pip install pillow numpy tqdm")
        sys.exit(1)
        
    main()