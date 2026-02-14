import numpy as np
import rasterio
import re
import os
import gc
import time
from pathlib import Path
import traceback

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
# Scikit-fuzzy and SciPy imports
import skfuzzy as fuzz
from scipy.ndimage import generate_binary_structure, binary_closing, binary_opening, binary_erosion
from sklearn.metrics.pairwise import euclidean_distances
# --- 2. WORKER LOGIC AND HELPER FUNCTIONS ---
# This section contains all the functions needed to process a single image.


# --- 2. WORKER LOGIC AND HELPER FUNCTIONS ---
def _get_kmeans_memberships(X, model):
    """Calculates pseudo-membership values for K-Means based on distance to centroids."""
    distances = euclidean_distances(X, model.cluster_centers_)
    # Invert and normalize distances to create a membership score
    memberships = 1.0 / (distances**2 + 1e-9)
    memberships /= np.sum(memberships, axis=1, keepdims=True)
    return memberships

def log_detailed_error(filename, config, error, details=None):
    """Logs detailed error information to a text file, including a full traceback."""
    log_dir = Path(config['output_dir']) / 'processing_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{filename}_error_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Error Log for file: {filename}\n")
        f.write(f"Timestamp: {pd.Timestamp.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Error Type: {type(error).__name__}\n")
        f.write(f"Error Message: {error}\n\n")
        if details:
            f.write("Additional Details:\n")
            for key, value in details.items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")
        f.write("Full Traceback:\n")
        f.write(traceback.format_exc())

def perform_kmeans_clustering(X_pca, config):
    """Perform K-means clustering with memory-efficient approach"""
    n_clusters = config['clustering']['n_clusters']
    n_samples = X_pca.shape[0]
    
    # Use MiniBatchKMeans for large datasets
    if n_samples > 100000:
        batch_size = min(10000, n_samples // 20)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=config['clustering']['kmeans']['n_init'],
            max_iter=config['clustering']['kmeans']['max_iter'],
            batch_size=batch_size,
            tol=config['clustering']['kmeans']['tolerance']
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=config['clustering']['kmeans']['n_init'],
            max_iter=config['clustering']['kmeans']['max_iter'],
            algorithm='lloyd',
            tol=config['clustering']['kmeans']['tolerance']
        )
    
    labels = kmeans.fit_predict(X_pca)
    
    # Calculate silhouette score on sample for large datasets
    if n_samples > 50000:
        sample_size = min(10000, n_samples)
        sample_idx = np.random.RandomState(42).choice(n_samples, sample_size, replace=False)
        sil_score = silhouette_score(X_pca[sample_idx], labels[sample_idx])
    else:
        sil_score = silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else 0
    
    return kmeans, labels, sil_score


def perform_gmm_clustering(X_pca, config):
    """Perform GMM clustering with memory management"""
    n_components = config['clustering']['n_clusters']
    
    # For large datasets, use a sample for fitting
    if X_pca.shape[0] > 200000:
        sample_size = min(50000, X_pca.shape[0])
        sample_idx = np.random.RandomState(42).choice(X_pca.shape[0], sample_size, replace=False)
        X_sample = X_pca[sample_idx]
    else:
        X_sample = X_pca
    
    # Enhanced parameters for stability
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=config['clustering']['gmm']['covariance_type'],
        random_state=42,
        n_init=config['clustering']['gmm']['n_init'],
        max_iter=config['clustering']['gmm']['max_iter'],
        tol=config['clustering']['gmm']['tolerance'],
        reg_covar=config['clustering']['gmm']['reg_covar']
    )
    
    # Fit on sample, predict on full dataset
    gmm.fit(X_sample)
    labels = gmm.predict(X_pca)
    labels = gmm.predict(X_pca)

# Check if clustering produced only one label (invalid result)
    if len(np.unique(labels)) < 2:
        return None, None, 0
    # Calculate silhouette score on sample
    if X_pca.shape[0] > 50000:
        sample_size = min(10000, X_pca.shape[0])
        sample_idx = np.random.RandomState(42).choice(X_pca.shape[0], sample_size, replace=False)
        sil_score = silhouette_score(X_pca[sample_idx], labels[sample_idx])
    else:
        sil_score = silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else 0
    
    return gmm, labels, sil_score


def perform_fuzzy_clustering(X, config):
    """
    Perform memory-efficient Fuzzy C-means clustering.
    
    Returns:
        cntr (ndarray): Cluster centers.
        u (ndarray): Membership matrix (shape: (n_clusters, n_samples)).
        labels (ndarray): Hard labels from membership argmax.
        sil_score (float): Silhouette score.
    """
    n_clusters = config['clustering']['n_clusters']
    m = config['clustering']['fuzzy']['m']
    max_iter = config['clustering']['fuzzy']['max_iter']
    error = config['clustering']['fuzzy']['error']
    
    n_samples = X.shape[0]

    # FCM expects shape (n_features, n_samples)
    data = X.T  
    
    # Run FCM clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=n_clusters, m=m, error=error, maxiter=max_iter, init=None, seed=42
    )

    # Hard labels = argmax of membership
    labels = np.argmax(u, axis=0).astype(np.int32)
    
    # Silhouette score (subsample if too big)
    if n_samples > 50000:
        sample_size = min(10000, n_samples)
        sample_idx = np.random.RandomState(42).choice(n_samples, sample_size, replace=False)
        sil_score = silhouette_score(X[sample_idx], labels[sample_idx])
    else:
        sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
    
    return cntr, u, labels, sil_score


# --- Other Helper Functions ---

def load_slope_mask(fid_number, config):
    if not config.get('slope_cleaning_enabled', False): return None
    slope_dir = Path(config['slope_dir'])
    slope_path = slope_dir / f"Slope_S1Grid_FID_{fid_number}.tif"
    if not slope_path.exists(): return None
    with rasterio.open(slope_path) as src:
        slope_data = src.read(1)
        return slope_data > config['slope_cleaning']['max_slope_degrees']

def identify_water_band(band_descriptions):
    for idx, desc in enumerate(band_descriptions):
        if 'vv' in desc.lower() and 'vh' not in desc.lower(): return idx
    return 0

def map_clusters_to_water_land(labels, original_data, band_descriptions):
    water_band_idx = identify_water_band(band_descriptions)
    cluster_stats = []
    for label in np.unique(labels):
        mask = labels == label
        if np.any(mask):
            mean_val = np.mean(original_data[mask, water_band_idx])
            cluster_stats.append({'label': label, 'mean': mean_val})
    if not cluster_stats: return np.zeros_like(labels, dtype=np.uint8), {}
    cluster_stats.sort(key=lambda x: x['mean'])
    water_label = cluster_stats[0]['label']
    water_land = np.zeros_like(labels, dtype=np.uint8)
    water_land[labels == water_label] = 1
    stats = {
        'water_percentage': np.mean(water_land) * 100,
        'water_pixels': int(np.sum(water_land)),
        'water_band_used': band_descriptions[water_band_idx],
        'water_label' : water_label
    }
    return water_land, stats


def apply_cleaning(image, slope_mask, config):
    water_binary = (image == 1)
    cleaned = water_binary.copy()
    
    if config.get('morphological_cleaning_enabled', True):
        iters = config['morphological_cleaning']['iterations']
        kernel = generate_binary_structure(2, 2)
        cleaned = binary_closing(cleaned, structure=kernel, iterations=iters)
        cleaned = binary_opening(cleaned, structure=kernel, iterations=iters)
        
        # Restore border pixels from original water_binary
        cleaned[0, :] = water_binary[0, :]    # Top row
        cleaned[-1, :] = water_binary[-1, :]  # Bottom row
        cleaned[:, 0] = water_binary[:, 0]    # Left column
        cleaned[:, -1] = water_binary[:, -1]  # Right column
    
    if slope_mask is not None and config.get('slope_cleaning_enabled', True):
        cleaned &= ~slope_mask
    
    final_image = np.full(image.shape, 0, dtype=np.int8)
    valid_mask = image != -1
    final_image[valid_mask & cleaned] = 1
    final_image[~valid_mask] = -1
    
    return final_image

def save_image(data, profile, path, tags):
    profile.update({'count': 1, 'dtype': 'int8', 'nodata': -1, 'compress': 'lzw'})
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(np.int8), 1)
        dst.update_tags(**tags)

def save_membership_array(memberships, mask, h, w, path):
    full = np.full((h * w, memberships.shape[1]), np.nan, dtype=np.float16)
    full[mask] = memberships
    np.save(path, full.reshape(h, w, -1))

def check_clear_bimodality_gmm(data: np.ndarray, separation_factor=2.0, min_weight=0.10):
    """
    Performs a multi-stage check for a CLEAR bimodal distribution.
    Returns (is_bimodal, reason_string).
    """
    if data.size < 500:
        return False, "Not enough valid data"
    data_reshaped = data.reshape(-1, 1)

    try:
        # 1. BIC Test
        gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=5).fit(data_reshaped)
        bic1 = gmm1.bic(data_reshaped)
        gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(data_reshaped)
        bic2 = gmm2.bic(data_reshaped)

        if bic2 >= bic1:
            return False, "Failed BIC test (unimodal is better fit)"

        # 2. Inspect GMM Components
        weights = gmm2.weights_
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())

        # 2a. Check for sufficient weight in both components
        if np.min(weights) < min_weight:
            return False, f"Failed weight balance (min weight {np.min(weights):.2f} < {min_weight})"

        # 2b. Check for sufficient separation between means
        mean_diff = abs(means[0] - means[1])
        avg_std = (stds[0] + stds[1]) / 2.0

        if mean_diff < separation_factor * avg_std:
            return False, f"Failed separation test (peaks too close)"

        return True, "Passed all checks"

    except Exception as e:
        return False, f"GMM convergence error: {e}"


def check_file_bimodality(filepath, bands_to_check=[5, 7, 10, 11], separation_factor=2.0, min_weight=0.10):
    """
    Check if file has bimodal distribution in any of the specified bands.
    Returns (is_bimodal, reason)
    """
    try:
        with rasterio.open(filepath) as src:
            for band_idx in bands_to_check:
                if band_idx > src.count:
                    continue
                    
                band_data = src.read(band_idx).astype(float)
                if src.nodata is not None:
                    valid_data = band_data[band_data != src.nodata].flatten()
                else:
                    valid_data = band_data.flatten()
                valid_data = valid_data[np.isfinite(valid_data)]

                is_bimodal, reason = check_clear_bimodality_gmm(valid_data, separation_factor, min_weight)

                if is_bimodal:
                    return True, f"Band {band_idx}: {reason}"
            
            return False, reason
    except Exception as e:
        return False, f"Error checking bimodality: {e}"
# --- Main Worker ---
def map_clusters_to_water_land_single_band(labels, single_band_data):
    """Maps clusters to water/land based on the mean of a single band's data."""
    cluster_stats = []
    unique_labels = np.unique(labels)
    
    # Ensure labels were actually generated
    if labels is None or len(unique_labels) == 0:
        return np.zeros_like(single_band_data, dtype=np.uint8), {'water_label': -1, 'cluster_stats': []}

    for label in unique_labels:
        mask = (labels == label)
        # Ensure the mask corresponds to actual data points used for clustering
        if np.any(mask) and mask.shape == single_band_data.shape:
             mean_val = np.mean(single_band_data[mask])
             cluster_stats.append({'label': label, 'mean': mean_val})
        elif np.any(mask): # Handle cases where labels might be shorter (if clustering sampled) - adjust if needed
             print(f"Warning: Label mask shape {mask.shape} differs from data shape {single_band_data.shape}. Skipping label {label}.")

    # Handle cases where clustering failed or only found one cluster robustly
    if len(cluster_stats) < 2: 
        water_land = np.zeros_like(labels, dtype=np.uint8) 
        water_label = -1 # Indicate failure or single cluster
        stats = {'water_percentage': 0, 'water_pixels': 0, 'water_label': water_label, 'cluster_stats': cluster_stats}
        return water_land, stats

    cluster_stats.sort(key=lambda x: x['mean'])
    water_label = cluster_stats[0]['label'] # Assume lower mean is water
    
    water_land = np.zeros_like(labels, dtype=np.uint8)
    water_land[labels == water_label] = 1
    
    stats = {
        'water_percentage': np.mean(water_land) * 100,
        'water_pixels': int(np.sum(water_land)),
        'water_label': water_label, # Return the identified water label index (0 or 1)
        'cluster_stats': cluster_stats
    }
    return water_land, stats
# In worker_functions.py

def process_single_image(filepath, config, semaphore):
    semaphore.acquire()
    try:
        start_time = time.time()
        filename_base = Path(filepath).stem
        error_details = {}
        
        try:
            # --- 1. DATA LOADING & BIMODALITY CHECK ---
            error_details['step'] = 'Data Loading & Bimodality Check'
            bimodal_band_idx = -1
            selected_band_data_1d = None # Will hold the valid data of the selected band
            
            with rasterio.open(filepath) as src:
                profile, h, w, descriptions = src.profile, src.height, src.width, src.descriptions
                descriptions = descriptions or [f'Band_{i+1}' for i in range(src.count)]
                nodata_val = src.nodata

                bands_to_check = config['bimodality_check']['bands_to_check']
                separation_factor = config['bimodality_check']['separation_factor']
                min_weight = config['bimodality_check']['min_weight']

                for band_idx in bands_to_check:
                    if band_idx > src.count: continue
                    
                    band_data_2d = src.read(band_idx).astype(float)
                    mask_2d = (band_data_2d == nodata_val) | np.isnan(band_data_2d) | (band_data_2d == 0) # Combine checks
                    band_data_2d[mask_2d] = np.nan 
                    
                    current_valid_data_1d = band_data_2d[~np.isnan(band_data_2d)]

                    # Use the bimodality check function from your other script
                    is_bimodal, reason = check_clear_bimodality_gmm(current_valid_data_1d, separation_factor, min_weight)
                    
                    if is_bimodal:
                        bimodal_band_idx = band_idx
                        # Need the original 2D data and mask for reconstructing the image later
                        selected_band_data_2d = band_data_2d 
                        selected_valid_mask_1d = ~np.isnan(band_data_2d.flatten()) # Mask matching full image size
                        # Prepare the 1D valid data FOR CLUSTERING
                        X_for_clustering = current_valid_data_1d.reshape(-1, 1) # Needs to be 2D for sklearn
                        print(f"  -> Found bimodal distribution in Band {bimodal_band_idx}. Using this band.")
                        break # Use the first bimodal band found
            
            # If no bimodal band was found after checking all specified bands
            if bimodal_band_idx == -1:
                return {'filename': filename_base, 'status': 'Skipped - No Bimodal Band Found', 'processing_time': time.time() - start_time}

            # --- 2. PREPROCESSING (Scaling Only) ---
            # PCA is skipped as we are using a single band
            error_details['step'] = 'Preprocessing'
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_for_clustering) # Scale the single band data
            pca_applied = False # Explicitly set PCA as not applied

            # --- 3. BASE CLUSTERING ALGORITHMS (On Single Band) ---
            error_details['step'] = 'Clustering'
            kmeans_model = gmm_model = fuzzy_u = None
            kmeans_labels = gmm_labels = fuzzy_labels = None
            
            try:
                # Use X_scaled (the single, scaled band) for clustering
                kmeans_model, kmeans_labels, _ = perform_kmeans_clustering(X_scaled, config) 
            except Exception as e:
                 print(f"  K-means failed: {e}") # Keep error handling
                 log_detailed_error(f"{filename_base}_kmeans", config, e, {'step': 'K-means clustering'})
            
            try:
                gmm_model, gmm_labels, _ = perform_gmm_clustering(X_scaled, config)
            except Exception as e:
                 print(f"  GMM failed: {e}")
                 log_detailed_error(f"{filename_base}_gmm", config, e, {'step': 'GMM clustering'})

            try:
                _, fuzzy_u, fuzzy_labels, _ = perform_fuzzy_clustering(X_scaled, config)
            except Exception as e:
                 print(f"  Fuzzy failed: {e}")
                 log_detailed_error(f"{filename_base}_fuzzy", config, e, {'step': 'Fuzzy clustering'})

            # --- 4. PROCESS INDIVIDUAL RESULTS ---
            error_details['step'] = 'Result Generation'
            results = {} 
            output_dir = Path(config['output_dir'])
            fid = (re.search(r'FID[_-]?(\d+)', filename_base, re.I) or [None, None])[1]
            slope_mask = load_slope_mask(fid, config) if fid else None
            
            # Use the 1D valid data from the selected band for mapping
            data_for_mapping = X_for_clustering.flatten() 

            for method, labels in [('kmeans', kmeans_labels), ('gmm', gmm_labels), ('fuzzy', fuzzy_labels)]:
                if labels is None: continue
                
                # Use the modified mapping function for single band data
                water_land, raw_stats = map_clusters_to_water_land_single_band(labels, data_for_mapping)
                
                # Reconstruct the full image using the valid mask for the selected band
                raw_img = np.full(selected_valid_mask_1d.shape, -1, dtype=np.int8) 
                raw_img[selected_valid_mask_1d] = water_land # Place results using the full mask
                raw_img = raw_img.reshape(h, w)
                
                # Modify output names to include the band index
                raw_dir = output_dir / f'{method}/raw_clusters'; raw_dir.mkdir(parents=True, exist_ok=True)
                save_image(raw_img, profile, raw_dir / f"{filename_base}_band{bimodal_band_idx}_{method}_raw.tif", {})

                cleaned_img = apply_cleaning(raw_img, slope_mask, config) # Pass the 2D raw_img now
                cleaned_dir = output_dir / f'{method}/cleaned_clusters'; cleaned_dir.mkdir(parents=True, exist_ok=True)
                save_image(cleaned_img, profile, cleaned_dir / f"{filename_base}_band{bimodal_band_idx}_{method}_cleaned.tif", {})
                
                cleaned_px = np.sum(cleaned_img == 1); valid_px = np.sum(cleaned_img != -1)
                cleaned_stats = {'water_percentage': cleaned_px / valid_px * 100 if valid_px > 0 else 0, 'water_pixels': int(cleaned_px)}
                results[method] = {'raw': raw_stats, 'cleaned': cleaned_stats}

            # --- 5. ENSEMBLE GENERATION (Using Single Band Memberships) ---
            error_details['step'] = 'Ensemble Generation'
            water_membership_list = []

            # Get aligned membership scores from each successful model (using X_scaled)
            if kmeans_model is not None:
                kmeans_mems = _get_kmeans_memberships(X_scaled, kmeans_model)
                _, raw_stats = map_clusters_to_water_land_single_band(kmeans_labels, data_for_mapping)
                if raw_stats['water_label'] != -1: # Check if mapping was successful
                   water_membership_list.append(kmeans_mems[:, raw_stats['water_label']])

            if gmm_model is not None:
                gmm_mems = gmm_model.predict_proba(X_scaled)
                _, raw_stats = map_clusters_to_water_land_single_band(gmm_labels, data_for_mapping)
                if raw_stats['water_label'] != -1:
                   water_membership_list.append(gmm_mems[:, raw_stats['water_label']])

            if fuzzy_u is not None:
                fuzzy_mems = fuzzy_u.T 
                _, raw_stats = map_clusters_to_water_land_single_band(fuzzy_labels, data_for_mapping)
                if raw_stats['water_label'] != -1:
                   water_membership_list.append(fuzzy_mems[:, raw_stats['water_label']])

            ensemble_labels = None
            ensemble_method = None
            if len(water_membership_list) >= 2:
                sum_of_memberships = np.sum(np.stack(water_membership_list), axis=0)
                # Ensure sum_of_memberships aligns with the clustered data (X_scaled length)
                if sum_of_memberships.shape[0] == X_scaled.shape[0]:
                    ensemble_labels = (sum_of_memberships > 2).astype(np.uint8) 
                    ensemble_method = 'ensemble_sum_thresh_2'
                else:
                    print(f"  Warning: Shape mismatch for ensemble sum ({sum_of_memberships.shape[0]}) vs data ({X_scaled.shape[0]}). Skipping ensemble.")

            # Process and save ensemble result if available
            if ensemble_labels is not None:
                # (Your existing code to reconstruct, save, and calculate stats for the ensemble image)
                # Make sure to modify output filenames to include band index
                # ...
                pass # Placeholder for your ensemble saving/stats code

            # --- 6. FINAL RETURN ---
            gc.collect()
            return {
                'filename': filename_base, 'status': 'Success', 'processing_time': time.time() - start_time,
                'pca_applied': pca_applied, # Will be False
                'band_used_for_analysis': bimodal_band_idx, # NEW: Record which band was used
                'slope_filtering_applied': slope_mask is not None,
                'fid': fid, 'file_size_mb': os.path.getsize(filepath) / (1024*1024), 'stats': results
            }
        
        except Exception as e:
            log_detailed_error(filename_base, config, e, details=error_details)
            return {'filename': filename_base, 'status': f'Failed: {e}', 'processing_time': time.time() - start_time}

    finally:
        semaphore.release()

