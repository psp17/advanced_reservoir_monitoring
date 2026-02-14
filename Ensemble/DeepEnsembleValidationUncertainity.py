import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import label, distance_transform_edt, binary_closing, binary_opening, generate_binary_structure
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
from tqdm import tqdm
import warnings
import gc
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, jaccard_score, cohen_kappa_score, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

# Import model architectures
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.base import SegmentationHead

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Checkpoints
DOFA_CHECKPOINT = '/home/arm/Desktop/ARM/Codes/DOFA/checkpoints/dofa_trainenc/best_model.pth'
SUMMIT_CHECKPOINT = '/home/arm/Desktop/ARM/Codes/SUMMIT-SAR/checkpoints/Summit+UPerNet/best_model.pth'
DEEPLABV3_CHECKPOINT = '/home/arm/Desktop/ARM/Codes/checkpoints_afterMMU (ResNet50 and DLV3+)/best_model.pth'
CROMA_CHECKPOINT = '/home/arm/Desktop/ARM/Codes/CROMA/checkpoints/CROMA+UPerNet/best_model.pth'
CROMA_FOUNDATION = '/home/arm/Desktop/ARM/Codes/CROMA/CROMA_base.pt'

NUM_CLASSES = 2
TILE_SIZE = 224
STRIDE = 112
SEED = 103

# Input Directories - NOW USING TIF DIRECTLY
INPUT_TIF_DIR = Path('/home/arm/Documents/ARM/Moisture_3N_Cluster_slope_3D/input')  # Source TIF files
GT_DIR = Path('/home/arm/Documents/ARM/Validation_Sen2indices/ClusterOutput/FinalGroundTruthWithS2')

# TIF Band Configuration (bands start from 1)
VH_BAND = 10  # Band 10 for VH
VV_BAND = 11  # Band 11 for VV

# Output Directory
OUTPUT_DIR = Path("/home/arm/Documents/ARM/Moisture_3N_Cluster_slope_3D/PNG/ensembleResult")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
UNCERTAINTY_DIR = OUTPUT_DIR / "uncertainty_maps"
UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
PROBABILITY_DIR = OUTPUT_DIR / "model_probabilities"
PROBABILITY_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# Create subfolders for each model
DOFA_PROB_DIR = PROBABILITY_DIR / "dofa"
DOFA_PROB_DIR.mkdir(parents=True, exist_ok=True)
SUMMIT_PROB_DIR = PROBABILITY_DIR / "summit"
SUMMIT_PROB_DIR.mkdir(parents=True, exist_ok=True)
DEEPLABV3_PROB_DIR = PROBABILITY_DIR / "deeplabv3"
DEEPLABV3_PROB_DIR.mkdir(parents=True, exist_ok=True)
CROMA_PROB_DIR = PROBABILITY_DIR / "croma"
CROMA_PROB_DIR.mkdir(parents=True, exist_ok=True)

# Normalization Constants
DOFA_MEAN = torch.tensor([166.36, 88.45], dtype=torch.float32)
DOFA_STD = torch.tensor([64.83, 43.07], dtype=torch.float32)
SUMMIT_MEAN = torch.tensor([166.36, 88.45, 127.41], dtype=torch.float32)
SUMMIT_STD = torch.tensor([64.83, 43.07, 53.95], dtype=torch.float32)

# Post-processing Config
POSTPROCESS_CONFIG = {
    'morphological_cleaning_enabled': True,
    'morphological_cleaning': {'iterations': 3},
    'mmu_filtering_enabled': True,
}

# Specific FIDs to process (FID, Quarter, Year)
SPECIFIC_FIDS = [
    (323, 3, 2018),
    (145, 2, 2017),
    (265, 3, 2015),
    (29, 3, 2024),
    (330, 1, 2016),
    (299, 3, 2019),
    (298, 3, 2019),
    (298, 2, 2019),
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_fid_info(filename):
    """Extract FID, Quarter, and Year from filename"""
    match = re.search(r'FID(\d+).*Y(\d{4}).*Q(\d)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(3)), int(match.group(2))
    
    match = re.search(r'id(\d+)_Q(\d)_(\d{4})', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    match = re.search(r'FID(\d+)_Q(\d)_(\d{4})', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    
    return None, None, None

def flush_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def read_vv_vh_from_tif(tif_path):
    """
    Read VV and VH bands directly from TIF file.
    Returns: vv_data, vh_data, crs, transform
    """
    try:
        with rasterio.open(tif_path) as src:
            # Check if bands exist
            if src.count < max(VV_BAND, VH_BAND):
                raise ValueError(f"TIF file has only {src.count} bands, need at least {max(VV_BAND, VH_BAND)}")
            
            # Read bands (rasterio uses 1-based indexing)
            vv_data = src.read(VV_BAND).astype('float32')
            vh_data = src.read(VH_BAND).astype('float32')
            
            crs = src.crs
            transform = src.transform
            
            print(f"    Read from TIF: VV (band {VV_BAND}), VH (band {VH_BAND})")
            print(f"    Shape: {vv_data.shape}, CRS: {crs}")
            
            return vv_data, vh_data, crs, transform
            
    except Exception as e:
        print(f"    ERROR reading TIF: {e}")
        raise

def identify_water_class(prediction, vv_data, vh_data):
    """Identify which class is water based on VV mean values"""
    unique_classes = np.unique(prediction)
    
    if len(unique_classes) == 1:
        stats = {
            'water_class_label': None,
            'water_percentage': 0.0,
            'water_pixels': 0,
            'mean_vv_class0': np.nan,
            'mean_vv_class1': np.nan
        }
        return np.zeros_like(prediction, dtype=np.uint8), stats
    
    class_stats = []
    for cls in unique_classes:
        mask = (prediction == cls)
        mean_vv = np.mean(vv_data[mask])
        class_stats.append({'label': cls, 'mean_vv': mean_vv})
    
    class_stats.sort(key=lambda x: x['mean_vv'])
    water_class = class_stats[0]['label']
    
    water_mask = (prediction == water_class).astype(np.uint8)
    
    stats = {
        'water_class_label': int(water_class),
        'water_percentage': float(np.mean(water_mask) * 100),
        'water_pixels': int(np.sum(water_mask)),
        'mean_vv_class0': float(np.mean(vv_data[prediction == 0])) if 0 in unique_classes else np.nan,
        'mean_vv_class1': float(np.mean(vv_data[prediction == 1])) if 1 in unique_classes else np.nan
    }
    
    return water_mask, stats

def apply_morphological_cleaning(image, config):
    """Apply morphological cleaning with edge preservation"""
    if not config.get('morphological_cleaning_enabled', True):
        return image
    
    water_binary = (image == 1)
    cleaned = water_binary.copy()
    
    iters = config['morphological_cleaning']['iterations']
    kernel = generate_binary_structure(2, 2)
    
    cleaned = binary_closing(cleaned, structure=kernel, iterations=iters)
    cleaned = binary_opening(cleaned, structure=kernel, iterations=iters)
    
    cleaned[0, :] = water_binary[0, :]
    cleaned[-1, :] = water_binary[-1, :]
    cleaned[:, 0] = water_binary[:, 0]
    cleaned[:, -1] = water_binary[:, -1]
    
    return cleaned.astype(np.uint8)

def apply_mmu_filtering(image, config):
    """Apply MMU filtering using 2-bin gap method"""
    if not config.get('mmu_filtering_enabled', False):
        return image
    
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(image == 1, structure=structure)
    
    if num_features < 2:
        return image
    
    region_sizes = np.bincount(labeled_array.ravel())[1:]
    max_size = region_sizes.max()
    
    num_bins = min(50, max(20, num_features // 5))
    hist, bin_edges = np.histogram(region_sizes, bins=num_bins)
    
    two_bin_gaps = []
    for i in range(len(hist) - 1):
        if i + 1 < len(hist) and hist[i] == 0 and hist[i + 1] == 0:
            is_start = (i == 0 or hist[i - 1] > 0)
            is_end = (i + 2 >= len(hist) or hist[i + 2] > 0)
            
            if is_start and is_end:
                start_value = bin_edges[i]
                end_value = bin_edges[i + 2]
                center = (start_value + end_value) / 2
                
                has_data_before = i > 0 and hist[i - 1] > 0
                has_data_after = i + 1 < len(hist) - 1 and hist[i + 2] > 0
                in_lower_range = center < max_size * 0.8
                
                if has_data_before and has_data_after and in_lower_range:
                    two_bin_gaps.append({'start_value': start_value, 'center': center})
    
    if len(two_bin_gaps) > 0:
        two_bin_gaps.sort(key=lambda x: x['center'])
        gap_threshold = two_bin_gaps[0]['start_value']
    else:
        gap_threshold = np.percentile(region_sizes, 20)
    
    p10 = np.percentile(region_sizes, 10)
    gap_threshold = max(gap_threshold, p10)
    
    size_threshold = max_size * 0.10
    final_threshold = max(gap_threshold, size_threshold)
    
    filtered_labels = labeled_array.copy()
    removed_count = 0
    for region_id in range(1, num_features + 1):
        if region_sizes[region_id - 1] < final_threshold:
            filtered_labels[filtered_labels == region_id] = 0
            removed_count += 1
    
    kept_count = num_features - removed_count
    
    if kept_count < 2:
        component_sizes = [(i+1, region_sizes[i]) for i in range(len(region_sizes))]
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        top_n = min(2, len(component_sizes))
        top_labels = [x[0] for x in component_sizes[:top_n]]
        
        filtered_labels = labeled_array.copy()
        for region_id in range(1, num_features + 1):
            if region_id not in top_labels:
                filtered_labels[filtered_labels == region_id] = 0
        
        kept_count = top_n
    
    if kept_count > 5:
        remaining_labels = np.unique(filtered_labels)[1:]
        remaining_sizes = [(lbl, np.sum(filtered_labels == lbl)) for lbl in remaining_labels]
        remaining_sizes.sort(key=lambda x: x[1], reverse=True)
        
        top_5_labels = [x[0] for x in remaining_sizes[:5]]
        for label_id in remaining_labels:
            if label_id not in top_5_labels:
                filtered_labels[filtered_labels == label_id] = 0
        
        kept_count = 5
    
    mmu_filtered = (filtered_labels > 0).astype(np.uint8)
    return mmu_filtered

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualization(vv_data, vh_data, ensemble_pred, uncertainty, model_probs, 
                        fid, year, quarter, output_path):
    """Create comprehensive visualization without GT"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Input data
    # VV
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(vv_data, cmap='gray')
    ax1.set_title('VV Band (Input)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # VH
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(vh_data, cmap='gray')
    ax2.set_title('VH Band (Input)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # VV/VH Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    ratio = np.divide(vv_data, vh_data + 1e-6)
    im3 = ax3.imshow(ratio, cmap='viridis')
    ax3.set_title('VV/VH Ratio', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Ensemble Prediction
    ax4 = fig.add_subplot(gs[0, 3])
    cmap_pred = ListedColormap(['#8B4513', '#1E90FF'])  # Brown for land, blue for water
    im4 = ax4.imshow(ensemble_pred, cmap=cmap_pred, vmin=0, vmax=1)
    ax4.set_title('Ensemble Prediction', fontsize=12, fontweight='bold')
    ax4.axis('off')
    legend_elements = [
        mpatches.Patch(facecolor='#8B4513', label='Land (0)'),
        mpatches.Patch(facecolor='#1E90FF', label='Water (1)')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Row 2: Individual model probabilities
    model_names = ['DOFA', 'SUMMIT', 'DeepLabV3+', 'CROMA']
    model_keys = ['dofa', 'summit', 'deeplabv3', 'croma']
    
    for idx, (name, key) in enumerate(zip(model_names, model_keys)):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(model_probs[key], cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title(f'{name} Water Prob.', fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', rotation=270, labelpad=15, fontsize=9)
    
    # Row 3: Uncertainty and statistics
    # Uncertainty Map
    ax9 = fig.add_subplot(gs[2, 0])
    im9 = ax9.imshow(uncertainty, cmap='hot', vmin=0, vmax=1)
    ax9.set_title('Uncertainty Map', fontsize=12, fontweight='bold')
    ax9.axis('off')
    cbar9 = plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
    cbar9.set_label('Uncertainty', rotation=270, labelpad=15, fontsize=9)
    
    # Uncertainty Histogram
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(uncertainty.ravel(), bins=50, color='red', alpha=0.7, edgecolor='black')
    ax10.set_xlabel('Uncertainty', fontsize=10)
    ax10.set_ylabel('Frequency', fontsize=10)
    ax10.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.axvline(uncertainty.mean(), color='blue', linestyle='--', linewidth=2, 
                 label=f'Mean: {uncertainty.mean():.3f}')
    ax10.legend(fontsize=9)
    
    # Model Agreement Map
    ax11 = fig.add_subplot(gs[2, 2])
    # Count how many models predict water (prob > 0.5)
    water_votes = np.zeros_like(ensemble_pred, dtype=float)
    for key in model_keys:
        water_votes += (model_probs[key] > 0.5).astype(float)
    im11 = ax11.imshow(water_votes, cmap='YlGnBu', vmin=0, vmax=4)
    ax11.set_title('Model Agreement (Votes)', fontsize=12, fontweight='bold')
    ax11.axis('off')
    cbar11 = plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
    cbar11.set_label('# Models Voting Water', rotation=270, labelpad=15, fontsize=9)
    
    # Statistics Panel
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    # Calculate statistics
    water_pixels = np.sum(ensemble_pred == 1)
    total_pixels = ensemble_pred.size
    water_pct = (water_pixels / total_pixels) * 100
    
    mean_uncertainty = uncertainty.mean()
    std_uncertainty = uncertainty.std()
    
    # Model probabilities stats
    prob_stats = []
    for name, key in zip(model_names, model_keys):
        prob_stats.append(f"{name}: {model_probs[key].mean():.3f} ± {model_probs[key].std():.3f}")
    
    stats_text = f"""
PREDICTION STATISTICS
{'='*30}

FID: {fid}
Year: {year}
Quarter: {quarter}

WATER COVERAGE
{'─'*30}
Total Pixels: {total_pixels:,}
Water Pixels: {water_pixels:,}
Water %: {water_pct:.2f}%

UNCERTAINTY
{'─'*30}
Mean: {mean_uncertainty:.4f}
Std: {std_uncertainty:.4f}
Min: {uncertainty.min():.4f}
Max: {uncertainty.max():.4f}

MODEL PROBABILITIES (Mean ± Std)
{'─'*30}
{chr(10).join(prob_stats)}

INPUT BANDS
{'─'*30}
VV Mean: {vv_data.mean():.2f}
VH Mean: {vh_data.mean():.2f}
VV/VH Ratio: {(vv_data.mean() / vh_data.mean()):.2f}
    """
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle(f'Deep Ensemble Water Detection - FID{fid} Y{year} Q{quarter}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Visualization saved: {output_path.name}")

def create_uncertainty_analysis_plot(all_uncertainties, output_path):
    """Create summary uncertainty analysis across all images"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ensemble Uncertainty Analysis - All Images', fontsize=16, fontweight='bold')
    
    # Flatten all uncertainties
    all_unc = np.concatenate([u.ravel() for u in all_uncertainties])
    
    # Histogram
    axes[0, 0].hist(all_unc, bins=100, color='red', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(all_unc.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {all_unc.mean():.3f}')
    axes[0, 0].axvline(np.median(all_unc), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(all_unc):.3f}')
    axes[0, 0].set_xlabel('Uncertainty', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Overall Uncertainty Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot([u.ravel() for u in all_uncertainties], vert=True)
    axes[0, 1].set_xlabel('Image Index', fontsize=11)
    axes[0, 1].set_ylabel('Uncertainty', fontsize=11)
    axes[0, 1].set_title('Uncertainty by Image', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # CDF
    sorted_unc = np.sort(all_unc)
    cdf = np.arange(1, len(sorted_unc) + 1) / len(sorted_unc)
    axes[1, 0].plot(sorted_unc, cdf, linewidth=2, color='darkred')
    axes[1, 0].set_xlabel('Uncertainty', fontsize=11)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=11)
    axes[1, 0].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0.95, color='gray', linestyle='--', alpha=0.5)
    
    # Statistics summary
    axes[1, 1].axis('off')
    stats_text = f"""
UNCERTAINTY STATISTICS
{'='*35}

Total Pixels: {len(all_unc):,}
Images Analyzed: {len(all_uncertainties)}

DISTRIBUTION
{'─'*35}
Mean:       {all_unc.mean():.4f}
Median:     {np.median(all_unc):.4f}
Std Dev:    {all_unc.std():.4f}
Min:        {all_unc.min():.4f}
Max:        {all_unc.max():.4f}

PERCENTILES
{'─'*35}
5th:        {np.percentile(all_unc, 5):.4f}
25th:       {np.percentile(all_unc, 25):.4f}
50th:       {np.percentile(all_unc, 50):.4f}
75th:       {np.percentile(all_unc, 75):.4f}
95th:       {np.percentile(all_unc, 95):.4f}

HIGH UNCERTAINTY
{'─'*35}
>0.5:       {(all_unc > 0.5).sum() / len(all_unc) * 100:.2f}%
>0.7:       {(all_unc > 0.7).sum() / len(all_unc) * 100:.2f}%
>0.9:       {(all_unc > 0.9).sum() / len(all_unc) * 100:.2f}%
    """
    
    axes[1, 1].text(0.1, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Uncertainty analysis saved: {output_path.name}")

# ============================================================================
# DEEP ENSEMBLE CLASS
# ============================================================================

class DeepEnsemble:
    """Deep Ensemble combining DOFA, CROMA, DeepLabV3+, and SUMMIT"""
    
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all four models"""
        print("Loading all models for deep ensemble...")
        
        # Import necessary modules for each model
        from dofa_v1 import vit_base_patch16
        import mae_model
        from use_croma import PretrainedCROMA
        
        # 1. Load DOFA
        print("  [1/4] Loading DOFA...")
        dofa_encoder = vit_base_patch16(img_size=TILE_SIZE)
        dofa_decoder = self._create_dofa_decoder()
        self.models['dofa'] = self._create_dofa_predictor(dofa_encoder, dofa_decoder)
        checkpoint = torch.load(DOFA_CHECKPOINT, map_location=self.device, weights_only=False)
        self.models['dofa'].encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.models['dofa'].decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.models['dofa'] = self.models['dofa'].to(self.device).eval()
        
        # 2. Load SUMMIT
        print("  [2/4] Loading SUMMIT...")
        summit_encoder = mae_model.mae_vit_base_patch16(img_size=TILE_SIZE)
        summit_decoder = self._create_summit_decoder()
        self.models['summit'] = self._create_summit_predictor(summit_encoder, summit_decoder)
        checkpoint = torch.load(SUMMIT_CHECKPOINT, map_location=self.device, weights_only=False)
        self.models['summit'].encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.models['summit'].decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.models['summit'] = self.models['summit'].to(self.device).eval()
        
        # 3. Load DeepLabV3+
        print("  [3/4] Loading DeepLabV3+...")
        self.models['deeplabv3'] = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=2,
            classes=NUM_CLASSES
        )
        checkpoint = torch.load(DEEPLABV3_CHECKPOINT, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.models['deeplabv3'].load_state_dict(state_dict)
        self.models['deeplabv3'] = self.models['deeplabv3'].to(self.device).eval()
        
        # 4. Load CROMA
        print("  [4/4] Loading CROMA...")
        croma_model = PretrainedCROMA(
            pretrained_path=CROMA_FOUNDATION,
            size='base',
            modality='SAR',
            image_resolution=TILE_SIZE
        )
        croma_decoder = self._create_croma_decoder()
        self.models['croma'] = self._create_croma_predictor(croma_model, croma_decoder)
        checkpoint = torch.load(CROMA_CHECKPOINT, map_location=self.device, weights_only=False)
        self.models['croma'].croma_model.load_state_dict(checkpoint['encoder_state_dict'])
        self.models['croma'].decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.models['croma'] = self.models['croma'].to(self.device).eval()
        
        print("✓ All models loaded successfully\n")
    
    def _create_dofa_decoder(self):
        """Create DOFA UPerNet decoder"""
        class DOFAUPerNet(nn.Module):
            def __init__(self, encoder_dim=768, decoder_channels=256, num_classes=2, patch_size=16, dropout=0.1):
                super().__init__()
                self.patch_size = patch_size
                encoder_channels = [3, encoder_dim, encoder_dim, encoder_dim, encoder_dim]
                
                self.feature_proj = nn.ModuleDict({
                    'block_3': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_5': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_7': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_11': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                })
                
                self.decoder = UPerNetDecoder(
                    encoder_channels=encoder_channels,
                    encoder_depth=4,
                    decoder_channels=decoder_channels,
                    use_norm="batchnorm",
                )
                
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels,
                    out_channels=num_classes,
                    activation=None,
                    kernel_size=1,
                    upsampling=4,
                )
                self.dropout = nn.Dropout2d(dropout)
            
            def reshape_vit_features(self, features, H, W):
                B, N, D = features.shape
                if N == (H * W) + 1:
                    features = features[:, 1:, :]
                return features.transpose(1, 2).reshape(B, D, H, W)
            
            def forward(self, features_dict, target_size, dummy_input=None):
                B, N, D = features_dict['block_11'].shape
                H = W = int(np.sqrt(N)) if N % int(np.sqrt(N)) == 0 else int(np.sqrt(N - 1))
                
                feat_3 = self.feature_proj['block_3'](self.reshape_vit_features(features_dict['block_3'], H, W))
                feat_5 = self.feature_proj['block_5'](self.reshape_vit_features(features_dict['block_5'], H, W))
                feat_7 = self.feature_proj['block_7'](self.reshape_vit_features(features_dict['block_7'], H, W))
                feat_11 = self.feature_proj['block_11'](self.reshape_vit_features(features_dict['block_11'], H, W))
                
                if dummy_input is None:
                    dummy_input = torch.zeros(B, 3, H, H, device=feat_3.device)
                else:
                    dummy_input = F.interpolate(dummy_input, size=(H, H), mode='bilinear', align_corners=False)
                
                features_list = [dummy_input, feat_3, feat_5, feat_7, feat_11]
                x = self.segmentation_head(self.dropout(self.decoder(features_list)))
                return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return DOFAUPerNet()
    
    def _create_dofa_predictor(self, encoder, decoder):
        """Create DOFA predictor wrapper"""
        class ViTFeatureExtractor:
            def __init__(self, model, hook_indices=[3, 5, 7, 11]):
                self.model = model
                self.hook_indices = hook_indices
                self.features = {}
                self.hooks = []
                for idx in hook_indices:
                    layer = model.blocks[idx]
                    self.hooks.append(layer.register_forward_hook(self._get_hook(f'block_{idx}')))

            def _get_hook(self, name):
                def hook(model, input, output):
                    self.features[name] = output
                return hook

            def clear(self):
                self.features = {}

            def remove_hooks(self):
                for h in self.hooks:
                    h.remove()
        
        class DOFAPredictor(nn.Module):
            def __init__(self, encoder, decoder, wavelengths=[3.75, 3.75]):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.wavelengths = wavelengths
                self.extractor = ViTFeatureExtractor(self.encoder, hook_indices=[3, 5, 7, 11])
                
            def forward(self, images):
                self.extractor.clear()
                _ = self.encoder(images, self.wavelengths)
                features_dict = self.extractor.features
                target_size = images.shape[-2:]
                logits = self.decoder(features_dict, target_size, dummy_input=images)
                return logits
        
        return DOFAPredictor(encoder, decoder)
    
    def _create_summit_decoder(self):
        """Create SUMMIT UPerNet decoder"""
        class SummitUPerNet(nn.Module):
            def __init__(self, encoder_dim=768, decoder_channels=256, num_classes=2, patch_size=16, dropout=0.1):
                super().__init__()
                self.patch_size = patch_size
                encoder_channels = [3, encoder_dim, encoder_dim, encoder_dim, encoder_dim]
                
                self.feature_proj = nn.ModuleDict({
                    'block_3': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_5': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_7': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'norm': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                })
                
                self.decoder = UPerNetDecoder(
                    encoder_channels=encoder_channels,
                    encoder_depth=4,
                    decoder_channels=decoder_channels,
                    use_norm="batchnorm",
                )
                
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels,
                    out_channels=num_classes,
                    activation=None,
                    kernel_size=1,
                    upsampling=4,
                )
                self.dropout = nn.Dropout2d(dropout)
            
            def reshape_vit_features(self, features, H, W):
                B, N, D = features.shape
                if N == (H * W) + 1:
                    features = features[:, 1:, :]
                return features.transpose(1, 2).reshape(B, D, H, W)
            
            def forward(self, features_dict, target_size, dummy_input=None):
                B, N, D = features_dict['norm'].shape
                H = W = int(np.sqrt(N)) if N % int(np.sqrt(N)) == 0 else int(np.sqrt(N - 1))
                
                feat_3 = self.feature_proj['block_3'](self.reshape_vit_features(features_dict['block_3'], H, W))
                feat_5 = self.feature_proj['block_5'](self.reshape_vit_features(features_dict['block_5'], H, W))
                feat_7 = self.feature_proj['block_7'](self.reshape_vit_features(features_dict['block_7'], H, W))
                feat_11 = self.feature_proj['norm'](self.reshape_vit_features(features_dict['norm'], H, W))
                
                if dummy_input is None:
                    dummy_input = torch.zeros(B, 3, H, H, device=feat_3.device)
                else:
                    dummy_input = F.interpolate(dummy_input, size=(H, H), mode='bilinear', align_corners=False)
                
                features_list = [dummy_input, feat_3, feat_5, feat_7, feat_11]
                x = self.segmentation_head(self.dropout(self.decoder(features_list)))
                return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return SummitUPerNet()
    
    def _create_summit_predictor(self, encoder, decoder):
        """Create SUMMIT predictor wrapper"""
        class ViTFeatureExtractor:
            def __init__(self, model):
                self.model = model
                self.features = {}
                self.hooks = []
                
                self.hooks.append(model.blocks[3].register_forward_hook(self._get_hook('block_3')))
                self.hooks.append(model.blocks[5].register_forward_hook(self._get_hook('block_5')))
                self.hooks.append(model.blocks[7].register_forward_hook(self._get_hook('block_7')))
                self.hooks.append(model.norm.register_forward_hook(self._get_hook('norm')))

            def _get_hook(self, name):
                def hook(model, input, output):
                    self.features[name] = output
                return hook

            def clear(self):
                self.features = {}

            def remove_hooks(self):
                for h in self.hooks:
                    h.remove()
        
        class SummitPredictor(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.extractor = ViTFeatureExtractor(self.encoder)
                
            def forward(self, images):
                self.extractor.clear()
                _ = self.encoder.forward_encoder(images, mask_ratio=0)
                features_dict = self.extractor.features
                target_size = images.shape[-2:]
                logits = self.decoder(features_dict, target_size, dummy_input=images)
                return logits
        
        return SummitPredictor(encoder, decoder)
    
    def _create_croma_decoder(self):
        """Create CROMA UPerNet decoder"""
        class CromaUPerNet(nn.Module):
            def __init__(self, encoder_dim=768, decoder_channels=256, num_classes=2, patch_size=8, dropout=0.1):
                super().__init__()
                self.patch_size = patch_size
                encoder_channels = [3, encoder_dim, encoder_dim, encoder_dim, encoder_dim]
                
                self.feature_proj = nn.ModuleDict({
                    'block_0': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_2': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'block_4': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                    'norm': nn.Conv2d(encoder_dim, encoder_dim, 1, bias=False),
                })
                
                self.decoder = UPerNetDecoder(
                    encoder_channels=encoder_channels,
                    encoder_depth=4,
                    decoder_channels=decoder_channels,
                    use_norm="batchnorm",
                )
                
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels,
                    out_channels=num_classes,
                    activation=None,
                    kernel_size=1,
                    upsampling=4,
                )
                self.dropout = nn.Dropout2d(dropout)
            
            def reshape_vit_features(self, features, H, W):
                B, N, D = features.shape
                return features.transpose(1, 2).reshape(B, D, H, W)
            
            def forward(self, features_dict, target_size, dummy_input=None):
                B, N, D = features_dict['norm'].shape
                H_feat = W_feat = int(np.sqrt(N))
                
                feat_1 = self.feature_proj['block_0'](self.reshape_vit_features(features_dict['block_0'], H_feat, W_feat))
                feat_3 = self.feature_proj['block_2'](self.reshape_vit_features(features_dict['block_2'], H_feat, W_feat))
                feat_5 = self.feature_proj['block_4'](self.reshape_vit_features(features_dict['block_4'], H_feat, W_feat))
                feat_norm = self.feature_proj['norm'](self.reshape_vit_features(features_dict['norm'], H_feat, W_feat))
                
                if dummy_input is None:
                    dummy_input = torch.zeros(B, 3, H_feat, H_feat, device=feat_1.device)
                else:
                    dummy_input = F.interpolate(dummy_input, size=(H_feat, H_feat), mode='bilinear', align_corners=False)
                
                features_list = [dummy_input, feat_1, feat_3, feat_5, feat_norm]
                x = self.segmentation_head(self.dropout(self.decoder(features_list)))
                return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return CromaUPerNet()
    
    def _create_croma_predictor(self, croma_model, decoder):
        """Create CROMA predictor wrapper"""
        class CromaFeatureExtractor:
            def __init__(self, croma_model):
                self.model = croma_model.s1_encoder
                self.features = {}
                self.hooks = []
                
                self.hooks.append(self.model.transformer.layers[0][1].register_forward_hook(self._get_hook('block_0')))
                self.hooks.append(self.model.transformer.layers[2][1].register_forward_hook(self._get_hook('block_2')))
                self.hooks.append(self.model.transformer.layers[4][1].register_forward_hook(self._get_hook('block_4')))
                self.hooks.append(self.model.transformer.norm_out.register_forward_hook(self._get_hook('norm')))

            def _get_hook(self, name):
                def hook(model, input, output):
                    self.features[name] = output
                return hook

            def clear(self):
                self.features = {}

            def remove_hooks(self):
                for h in self.hooks:
                    h.remove()
        
        class CromaPredictor(nn.Module):
            def __init__(self, croma_model, decoder):
                super().__init__()
                self.croma_model = croma_model
                self.decoder = decoder
                self.extractor = CromaFeatureExtractor(self.croma_model)
                
            def forward(self, images):
                self.extractor.clear()
                attn_bias = self.croma_model.attn_bias.to(images.device)
                _ = self.croma_model.s1_encoder(images, attn_bias=attn_bias)
                features_dict = self.extractor.features
                target_size = images.shape[-2:]
                
                dummy = F.pad(images, (0,0,0,0,0,1), "constant", 0)
                logits = self.decoder(features_dict, target_size, dummy_input=dummy)
                return logits
        
        return CromaPredictor(croma_model, decoder)
    
    def normalize_input(self, vv, vh, model_name):
        """Normalize input based on model requirements"""
        if model_name == 'dofa' or model_name == 'croma' or model_name == 'deeplabv3':
            # 2 channels: VV, VH
            image_np = np.stack([vv, vh], axis=0)
            mean = DOFA_MEAN.to(self.device).view(2, 1, 1)
            std = DOFA_STD.to(self.device).view(2, 1, 1)
        elif model_name == 'summit':
            # 3 channels: VV, VH, Avg
            avg = (vv + vh) / 2.0
            image_np = np.stack([vv, vh, avg], axis=0)
            mean = SUMMIT_MEAN.to(self.device).view(3, 1, 1)
            std = SUMMIT_STD.to(self.device).view(3, 1, 1)
        
        image_tensor = torch.from_numpy(image_np).to(self.device)
        return (image_tensor - mean) / std
    
    @torch.inference_mode()
    def predict_single_model(self, vv_data, vh_data, model_name):
        """Predict with a single model and return water probabilities"""
        H_orig, W_orig = vv_data.shape
        
        image_tensor = self.normalize_input(vv_data, vh_data, model_name)
        
        # Handle small images
        if H_orig < TILE_SIZE or W_orig < TILE_SIZE:
            image_tensor = image_tensor.unsqueeze(0)
            resized_image = F.interpolate(image_tensor, size=(TILE_SIZE, TILE_SIZE), 
                                        mode='bilinear', align_corners=False)
            logits = self.models[model_name](resized_image)
            probs = torch.softmax(logits, dim=1)
            orig_size_probs = F.interpolate(probs, size=(H_orig, W_orig), 
                                          mode='bilinear', align_corners=False)
            
            # Return water probability (class 1)
            water_prob = orig_size_probs.squeeze(0)[1].cpu().numpy()
            del image_tensor, resized_image, logits, probs, orig_size_probs
            return water_prob
        
        # Sliding window for large images
        pad_w = (STRIDE - (W_orig - TILE_SIZE) % STRIDE) % STRIDE
        pad_h = (STRIDE - (H_orig - TILE_SIZE) % STRIDE) % STRIDE
        
        padded_image = F.pad(image_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='replicate')
        _, C, H_padded, W_padded = padded_image.shape
        
        prediction_sum = torch.zeros((NUM_CLASSES, H_padded, W_padded), 
                                    dtype=torch.float32, device=self.device)
        pixel_counts = torch.zeros((1, H_padded, W_padded), 
                                  dtype=torch.float32, device=self.device)
        
        for y in range(0, H_padded - TILE_SIZE + 1, STRIDE):
            for x in range(0, W_padded - TILE_SIZE + 1, STRIDE):
                tile = padded_image[:, :, y:y+TILE_SIZE, x:x+TILE_SIZE]
                logits = self.models[model_name](tile)
                probs = torch.softmax(logits, dim=1)
                prediction_sum[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += probs.squeeze(0)
                pixel_counts[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += 1
        
        avg_probs = prediction_sum / (pixel_counts + 1e-6)
        water_prob = avg_probs[1, 0:H_orig, 0:W_orig].cpu().numpy()
        
        del padded_image, prediction_sum, pixel_counts, avg_probs
        flush_memory()
        
        return water_prob
    
    def predict_ensemble_with_uncertainty(self, vv_data, vh_data):
        """
        Predict with all models and calculate uncertainty using paper's approach.
        Returns: ensemble_prediction, uncertainty_map, individual_probabilities
        """
        # Get predictions from all models
        model_probs = {}
        for model_name in ['dofa', 'summit', 'deeplabv3', 'croma']:
            print(f"      - {model_name.upper()}")
            model_probs[model_name] = self.predict_single_model(vv_data, vh_data, model_name)
            flush_memory()
        
        # Stack probabilities: shape (n_models, H, W)
        prob_stack = np.stack([model_probs[name] for name in ['dofa', 'summit', 'deeplabv3', 'croma']], axis=0)
        n_models = prob_stack.shape[0]
        
        # Calculate ensemble mean probability
        mean_prob = np.mean(prob_stack, axis=0)
        
        # Generate ensemble prediction (threshold at 0.5)
        ensemble_pred = (mean_prob > 0.5).astype(np.uint8)
        
        # Calculate uncertainty using equations from the paper
        sum_probs = np.sum(prob_stack, axis=0)
        
        # Initialize uncertainty map
        uncertainty = np.zeros_like(mean_prob)
        
        # For pixels predicted as non-water (label = 0)
        mask_non_water = (mean_prob < 0.5)
        uncertainty[mask_non_water] = (2.0 / n_models) * sum_probs[mask_non_water]
        
        # For pixels predicted as water (label = 1)
        mask_water = (mean_prob >= 0.5)
        uncertainty[mask_water] = 2.0 - (2.0 / n_models) * sum_probs[mask_water]
        
        # Clip uncertainty to [0, 1]
        uncertainty = np.clip(uncertainty, 0, 1)
        
        return ensemble_pred, uncertainty, model_probs

# ============================================================================
# VALIDATION METRICS
# ============================================================================

def calculate_pixel_metrics(pred, gt, cloud_mask):
    """Calculate pixel-level metrics excluding cloud pixels"""
    valid_mask = ~cloud_mask
    pred_valid = pred[valid_mask].flatten()
    gt_valid = gt[valid_mask].flatten()
    
    if len(pred_valid) == 0:
        return {
            'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan,
            'f1_score': np.nan, 'iou': np.nan, 'kappa': np.nan,
            'true_positives': 0, 'false_positives': 0,
            'true_negatives': 0, 'false_negatives': 0,
            'valid_pixels': 0
        }
    
    tn, fp, fn, tp = confusion_matrix(gt_valid, pred_valid, labels=[0, 1]).ravel()
    
    return {
        'accuracy': accuracy_score(gt_valid, pred_valid),
        'precision': precision_score(gt_valid, pred_valid, zero_division=0),
        'recall': recall_score(gt_valid, pred_valid, zero_division=0),
        'f1_score': f1_score(gt_valid, pred_valid, zero_division=0),
        'iou': jaccard_score(gt_valid, pred_valid, zero_division=0),
        'kappa': cohen_kappa_score(gt_valid, pred_valid),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'valid_pixels': len(pred_valid)
    }

def calculate_hausdorff_distance(pred, gt):
    """Calculate Hausdorff distance between boundaries"""
    pred_boundary = np.logical_xor(pred, distance_transform_edt(pred) > 1)
    gt_boundary = np.logical_xor(gt, distance_transform_edt(gt) > 1)
    
    pred_coords = np.argwhere(pred_boundary)
    gt_coords = np.argwhere(gt_boundary)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan
    
    hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
    
    return max(hd1, hd2)

def calculate_boundary_iou(pred, gt, boundary_width=5):
    """Calculate IoU of boundary regions"""
    pred_dist = distance_transform_edt(~pred.astype(bool))
    gt_dist = distance_transform_edt(~gt.astype(bool))
    
    pred_boundary = (pred_dist <= boundary_width) & (pred_dist > 0)
    gt_boundary = (gt_dist <= boundary_width) & (gt_dist > 0)
    
    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()
    
    return intersection / union if union > 0 else np.nan

def calculate_area_metrics(pred, gt, pixel_size_m2=100):
    """Calculate area-based metrics"""
    pred_area = pred.sum() * pixel_size_m2
    gt_area = gt.sum() * pixel_size_m2
    
    area_diff = pred_area - gt_area
    area_diff_pct = (area_diff / gt_area * 100) if gt_area > 0 else np.nan
    
    return {
        'pred_area_m2': float(pred_area),
        'gt_area_m2': float(gt_area),
        'area_diff_m2': float(area_diff),
        'area_diff_pct': float(area_diff_pct)
    }

def validate_image_pair(pred, gt_path):
    """Validate a single image pair with cloud-aware logic"""
    with rasterio.open(gt_path) as src:
        gt = src.read(1).astype(np.uint8)
    
    cloud_mask = (gt == 255)
    cloud_percentage = (cloud_mask.sum() / cloud_mask.size) * 100
    gt_binary = (gt == 1).astype(np.uint8)
    
    if pred.shape != gt_binary.shape:
        print(f"Warning: Shape mismatch - pred: {pred.shape}, gt: {gt_binary.shape}")
        return None
    
    pixel_metrics = calculate_pixel_metrics(pred, gt_binary, cloud_mask)
    
    results = {
        'has_clouds': cloud_percentage > 0,
        'cloud_percentage': cloud_percentage,
        **pixel_metrics
    }
    
    if cloud_percentage > 1.0:
        results.update({
            'hausdorff_distance': np.nan,
            'boundary_iou': np.nan,
            'pred_area_m2': np.nan,
            'gt_area_m2': np.nan,
            'area_diff_m2': np.nan,
            'area_diff_pct': np.nan
        })
        return results
    
    results['hausdorff_distance'] = calculate_hausdorff_distance(pred, gt_binary)
    results['boundary_iou'] = calculate_boundary_iou(pred, gt_binary)
    
    area_metrics = calculate_area_metrics(pred, gt_binary)
    results.update(area_metrics)
    
    return results

def save_prediction_tiff(prediction, reference_crs, reference_transform, output_path, height, width):
    """Save prediction as GeoTIFF with proper georeferencing"""
    try:
        # Ensure output shape matches
        if prediction.shape != (height, width):
            print(f"      WARNING: Shape mismatch - pred: {prediction.shape}, expected: ({height}, {width})")
            return False
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw',
            'nodata': None,
            'height': height,
            'width': width,
            'crs': reference_crs,
            'transform': reference_transform
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction.astype(np.uint8), 1)
        
        return True
    except Exception as e:
        print(f"      ERROR saving prediction: {e}")
        return False

def save_uncertainty_tiff(uncertainty, reference_crs, reference_transform, output_path, height, width):
    """Save uncertainty map as GeoTIFF (uint16, scaled 0-10000 for 0.00-100.00%)"""
    try:
        if uncertainty.shape != (height, width):
            print(f"      WARNING: Shape mismatch - uncertainty: {uncertainty.shape}, expected: ({height}, {width})")
            return False
        
        uncertainty_scaled = np.round(uncertainty * 10000).astype(np.uint16)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'count': 1,
            'compress': 'lzw',
            'nodata': None,
            'height': height,
            'width': width,
            'crs': reference_crs,
            'transform': reference_transform
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(uncertainty_scaled, 1)
            dst.update_tags(1, 
                scale_factor='0.0001',
                description='Uncertainty scaled by 10000. Divide by 10000 to get 0-1 range, or by 100 for percentage.',
                units='scaled_percentage',
                example='6668 = 66.68% uncertainty')
        
        return True
    except Exception as e:
        print(f"      ERROR saving uncertainty: {e}")
        return False

def save_probability_tiff(probability, reference_crs, reference_transform, output_path, model_name, height, width):
    """Save probability map as GeoTIFF (uint16, scaled 0-10000 for 0.00-100.00%)"""
    try:
        if probability.shape != (height, width):
            print(f"      WARNING: Shape mismatch - {model_name} prob: {probability.shape}, expected: ({height}, {width})")
            return False
        
        probability_scaled = np.round(probability * 10000).astype(np.uint16)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint16',
            'count': 1,
            'compress': 'lzw',
            'nodata': None,
            'height': height,
            'width': width,
            'crs': reference_crs,
            'transform': reference_transform
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(probability_scaled, 1)
            dst.update_tags(1,
                model_name=model_name,
                scale_factor='0.0001',
                description='Water probability scaled by 10000. Divide by 10000 to get 0-1 range, or by 100 for percentage.',
                units='scaled_percentage',
                example='6668 = 66.68% water probability')
        
        return True
    except Exception as e:
        print(f"      ERROR saving {model_name} probability: {e}")
        return False

# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("DEEP ENSEMBLE PREDICTION + VISUALIZATION (TIF DIRECT)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Models: DOFA + SUMMIT + DeepLabV3+ + CROMA")
    print(f"Input: VV=Band{VV_BAND}, VH=Band{VH_BAND} from TIF files")
    print(f"Post-processing: Morphological Cleaning + MMU Filtering")
    print(f"Strategy: Process SPECIFIC FIDs with full visualization")
    print(f"Specific FIDs requested: {len(SPECIFIC_FIDS)}")
    for fid, q, y in SPECIFIC_FIDS:
        print(f"  - FID{fid} Q{q} Y{y}")
    
    # Load ensemble
    print("\n[1/6] Loading Deep Ensemble...")
    ensemble = DeepEnsemble(device)
    
    # Discover ALL TIF files
    print("\n[2/6] Discovering ALL TIF files...")
    all_tif_files = {}
    
    for tif_file in INPUT_TIF_DIR.glob("*.tif"):
        fid, quarter, year = extract_fid_info(tif_file.name)
        if fid:
            key = (fid, year, quarter)
            all_tif_files[key] = tif_file
    
    print(f"  Found {len(all_tif_files)} TIF files")
    
    # Filter to only specific FIDs
    tif_pairs = []
    specific_fids_set = set(SPECIFIC_FIDS)
    
    for key, tif_path in all_tif_files.items():
        fid, year, quarter = key
        if (fid, quarter, year) in specific_fids_set:
            tif_pairs.append({
                'fid': fid,
                'year': year,
                'quarter': quarter,
                'tif': tif_path
            })
    
    print(f"  Found {len(tif_pairs)} TIF files matching specific FIDs")
    
    # Discover ground truth files
    print("\n[3/6] Discovering ground truth files (optional)...")
    gt_files = {}
    for gt_path in GT_DIR.glob("*.tif"):
        fid, quarter, year = extract_fid_info(gt_path.name)
        if fid:
            key = (fid, year, quarter)
            gt_files[key] = gt_path
    
    print(f"  Found {len(gt_files)} ground truth files")
    
    # Map which pairs have GT
    pairs_with_gt = set(gt_files.keys())
    pairs_for_validation = []
    pairs_without_gt = []
    
    for pair in tif_pairs:
        key = (pair['fid'], pair['year'], pair['quarter'])
        if key in pairs_with_gt:
            pair['gt'] = gt_files[key]
            pairs_for_validation.append(pair)
        else:
            pairs_without_gt.append(pair)
    
    print(f"  {len(pairs_for_validation)} pairs WITH ground truth (will validate)")
    print(f"  {len(pairs_without_gt)} pairs WITHOUT ground truth (prediction + viz only)")
    
    if len(tif_pairs) == 0:
        print("No TIF files found!")
        return
    
    # Run predictions on ALL images
    print("\n[4/6] Running ensemble prediction + visualization...")
    
    image_results = []
    fid_aggregates = defaultdict(list)
    all_uncertainties = []
    
    all_pairs = pairs_for_validation + pairs_without_gt
    
    for pair in tqdm(all_pairs, desc="Processing"):
        try:
            has_gt = 'gt' in pair
            status = "WITH GT" if has_gt else "NO GT"
            print(f"\n  Processing FID{pair['fid']} Y{pair['year']} Q{pair['quarter']} ({status})")
            print(f"    TIF file: {pair['tif'].name}")
            
            # Read VV/VH data directly from TIF bands
            vv_data, vh_data, crs, transform = read_vv_vh_from_tif(pair['tif'])
            
            # Ensemble prediction with uncertainty
            print(f"    Running ensemble...")
            ensemble_pred, uncertainty, model_probs = ensemble.predict_ensemble_with_uncertainty(
                vv_data, vh_data
            )
            
            # Store for analysis
            all_uncertainties.append(uncertainty)
            
            # Identify water class
            water_mask, water_stats = identify_water_class(ensemble_pred, vv_data, vh_data)
            
            # Post-processing
            print(f"    Applying post-processing...")
            cleaned_mask = apply_morphological_cleaning(water_mask, POSTPROCESS_CONFIG)
            final_mask = apply_mmu_filtering(cleaned_mask, POSTPROCESS_CONFIG)
            
            height, width = vv_data.shape
            
            # Create visualization
            print(f"    Creating visualization...")
            viz_filename = f"FID{pair['fid']:03d}_Y{pair['year']}_Q{pair['quarter']}_visualization.png"
            viz_path = VISUALIZATION_DIR / viz_filename
            create_visualization(vv_data, vh_data, final_mask, uncertainty, model_probs,
                               pair['fid'], pair['year'], pair['quarter'], viz_path)
            
            # Save prediction
            pred_filename = f"FID{pair['fid']:03d}_Y{pair['year']}_Q{pair['quarter']}_ensemble_prediction.tif"
            pred_path = PREDICTIONS_DIR / pred_filename
            print(f"    Saving prediction...")
            save_prediction_tiff(final_mask, crs, transform, pred_path, height, width)
            
            # Save uncertainty map
            uncertainty_filename = f"FID{pair['fid']:03d}_Y{pair['year']}_Q{pair['quarter']}_uncertainty.tif"
            uncertainty_path = UNCERTAINTY_DIR / uncertainty_filename
            print(f"    Saving uncertainty...")
            save_uncertainty_tiff(uncertainty, crs, transform, uncertainty_path, height, width)
            
            # Save individual model probabilities
            model_dirs = {
                'dofa': DOFA_PROB_DIR,
                'summit': SUMMIT_PROB_DIR,
                'deeplabv3': DEEPLABV3_PROB_DIR,
                'croma': CROMA_PROB_DIR
            }
            
            print(f"    Saving model probabilities...")
            for model_name, prob_map in model_probs.items():
                prob_filename = f"FID{pair['fid']:03d}_Y{pair['year']}_Q{pair['quarter']}_probability.tif"
                prob_path = model_dirs[model_name] / prob_filename
                save_probability_tiff(prob_map, crs, transform, prob_path, model_name.upper(), height, width)
            
            # Validate ONLY if ground truth exists
            if has_gt:
                print(f"    Validating against ground truth...")
                metrics = validate_image_pair(final_mask, pair['gt'])
                
                if metrics is not None:
                    result = {
                        'fid': pair['fid'],
                        'year': pair['year'],
                        'quarter': pair['quarter'],
                        'filename': f"FID{pair['fid']}_Y{pair['year']}_Q{pair['quarter']}",
                        'mean_uncertainty': float(np.mean(uncertainty)),
                        'std_uncertainty': float(np.std(uncertainty)),
                        **metrics
                    }
                    image_results.append(result)
                    fid_aggregates[pair['fid']].append(metrics)
            else:
                print(f"    ✓ Prediction & visualization saved (no GT)")
            
            del ensemble_pred, uncertainty, water_mask, cleaned_mask, final_mask, vv_data, vh_data
            if len(all_pairs) % 5 == 0:
                flush_memory()
                
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create uncertainty analysis
    print("\n[5/6] Creating uncertainty analysis...")
    if len(all_uncertainties) > 0:
        analysis_path = VISUALIZATION_DIR / "uncertainty_analysis_summary.png"
        create_uncertainty_analysis_plot(all_uncertainties, analysis_path)
    
    # Calculate and save metrics
    print("\n[6/6] Saving results...")
    
    if len(image_results) > 0:
        df_images = pd.DataFrame(image_results)
        
        # FID-level results
        fid_results = []
        for fid, metrics_list in fid_aggregates.items():
            df_temp = pd.DataFrame(metrics_list)
            
            fid_result = {
                'fid': fid,
                'n_images': len(metrics_list),
                'mean_accuracy': df_temp['accuracy'].mean(),
                'std_accuracy': df_temp['accuracy'].std(),
                'mean_precision': df_temp['precision'].mean(),
                'std_precision': df_temp['precision'].std(),
                'mean_recall': df_temp['recall'].mean(),
                'std_recall': df_temp['recall'].std(),
                'mean_f1_score': df_temp['f1_score'].mean(),
                'std_f1_score': df_temp['f1_score'].std(),
                'mean_iou': df_temp['iou'].mean(),
                'std_iou': df_temp['iou'].std(),
            }
            fid_results.append(fid_result)
        
        df_fids = pd.DataFrame(fid_results)
        
        # Save Excel
        excel_path = OUTPUT_DIR / 'ensemble_validation_results.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_images.to_excel(writer, sheet_name='Image_Level_Metrics', index=False)
            df_fids.to_excel(writer, sheet_name='FID_Level_Metrics', index=False)
        
        print(f"  Saved: {excel_path}")
    
    # Save summary
    summary_path = OUTPUT_DIR / 'prediction_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DEEP ENSEMBLE PREDICTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Predictions: {len(all_pairs)}\n")
        f.write(f"  - With GT (validated): {len(pairs_for_validation)}\n")
        f.write(f"  - Without GT: {len(pairs_without_gt)}\n\n")
        f.write(f"Visualizations: {VISUALIZATION_DIR}\n")
        f.write(f"Predictions: {PREDICTIONS_DIR}\n")
        f.write(f"Uncertainties: {UNCERTAINTY_DIR}\n")
        f.write(f"Probabilities: {PROBABILITY_DIR}\n")
    
    print(f"  Saved: {summary_path}")
    
    # Console summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTotal Processed: {len(all_pairs)}")
    print(f"  ├─ With GT: {len(pairs_for_validation)}")
    print(f"  └─ Without GT: {len(pairs_without_gt)}")
    print("\n" + "=" * 80)
    print("ALL OUTPUTS SAVED")
    print("=" * 80)
    print(f"Visualizations: {VISUALIZATION_DIR}")
    print(f"  └─ {len(all_pairs)} PNG files")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print(f"  └─ {len(all_pairs)} TIF files")
    print(f"Uncertainties: {UNCERTAINTY_DIR}")
    print(f"  └─ {len(all_pairs)} TIF files (uint16)")
    print(f"Probabilities: {PROBABILITY_DIR}")
    print(f"  └─ {len(all_pairs) * 4} TIF files (4 models, uint16)")
    print("=" * 80)

if __name__ == "__main__":
    main()