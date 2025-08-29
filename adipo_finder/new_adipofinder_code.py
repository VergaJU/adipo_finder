import squidpy as sq
import scanpy as sc
import pandas as pd 
import numpy as np 
from adipo_finder import utils as seg_utils
from adipo_finder import segmentation as seg
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from skimage import measure, morphology, segmentation, feature
from scipy.ndimage import distance_transform_edt, label, binary_dilation
from sklearn.datasets import make_blobs
#just load one for now
from PIL import Image
from skimage.measure import regionprops
from skimage.morphology import disk
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import random
import os


def shuffle_labels(labeled_image):
    '''
    Shuffles the labels of a labeled image, making it more
    appealing to plot, since segments next to each other have
    more diverse label indices, and thereby more diverse colors.
    '''
    # Shuffle label values (excluding 0 which is background)
    labels = np.unique(labeled_image)
    labels = labels[labels != 0]
    shuffled_labels = np.random.permutation(labels)

    # Create a mapping
    label_map = {old: new for old, new in zip(labels, shuffled_labels)}
    shuffled_labeled_image = np.copy(labeled_image)
    for old, new in label_map.items():
        shuffled_labeled_image[labeled_image == old] = new
    return shuffled_labeled_image

#
def process_ground_truth_image(base_path, id):
    '''
    Generates a segmented ground truth image from the binary.
    id - string like "ROI001_02454_ROI_1"
    '''
    #load:
    img_tmp = Image.open(base_path + id + " step 3.tif")
    raw_gt_img = np.array(img_tmp)
    #segment:
    labeled_image = measure.label(raw_gt_img)
    #shuffle labels for better presentation - no good if segments next to each other have very similar indices
    shuffled_labeled_image = shuffle_labels(labeled_image)
    #write image to file
    img = Image.fromarray(shuffled_labeled_image.astype(np.uint16))
    img.save(base_path + id + " step 4.png")
    #also create a plot and save it - in the plot we use a color scheme that makes it easier to see segments
    plt.figure(figsize=(8, 8))
    plt.imshow(shuffled_labeled_image, cmap='nipy_spectral') 
    plt.title("Shuffled final")
    plt.axis("off") 
    plt.tight_layout()
    #plt.show()
    plt.savefig(base_path + id + " ground_truth_plot.png", dpi=300, bbox_inches="tight")  # save instead of show
    plt.close()#prevents image from being shown in notebooks, annoying if you run many in a loop
    

def get_seg_colormap(cmap=plt.cm.nipy_spectral, minval=0.2, maxval=1.0, n=5000):
    """Creates a colormap with black for no segment and bright colors for the rest"""
    new_cmap = plt.cm.get_cmap(cmap)
    new_colors = new_cmap(np.linspace(minval, maxval, n))
    colmap = mcolors.ListedColormap(new_colors)
    colors = colmap(np.arange(colmap.N))  # shape (n, 4)

    # Prepend black for background - black is the background
    colors = np.vstack(([0, 0, 0, 1], colors))
    colmap_with_black = mcolors.ListedColormap(colors)
    return colmap_with_black

def plot_pred_vs_gt(bin_mask_image, segmented_unfiltered, pred_image, gt_image):
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    # Panel 1: original mask
    axes[0].imshow(bin_mask_image, cmap='gray')
    axes[0].set_title("Binary mask")

    
    
    # Panel 2: unfiltered segmentation on mask
    # Make truncated colormap
    colmap = truncate_colormap(plt.cm.nipy_spectral)

    # Convert to array of RGBA colors
    colors = colmap(np.arange(colmap.N))  # shape (n, 4)

    # Prepend black for background
    colors = np.vstack(([0, 0, 0, 1], colors))
    colmap_with_black = mcolors.ListedColormap(colors)


def preprocess_input_image(seg_input_img, min_distance_seg_init = 30, min_island_area = 0):
    bin_mask_image = seg_utils.Preprocessing.segmentation_to_binary(seg_input_img)
    #optionally clean up the image a bit - this will give larger segments in the ocean, but may
    #also make it harder to capture adipocytes on the shores of the continents
    cleaned_mask_image = bin_mask_image.copy()
    if min_island_area > 0:
        labeled_image = measure.label(bin_mask_image, connectivity=2) #2 means 8-connectivity, so also connected via corners
        props = measure.regionprops(labeled_image, intensity_image=bin_mask_image)
    
        for prop in props:
            label_id = prop.label
            if label_id == 0:
                continue  # skip background

            if prop.area < min_island_area:
                cleaned_mask_image[labeled_image == label_id] = 0

    inverted_image = seg_utils.Preprocessing.invert_image(cleaned_mask_image)
    # perform opening
    structuring_element = morphology.disk(4)
    # structuring_element = np.ones((window, window))
    inverted_opened_img = morphology.opening(inverted_image, footprint=structuring_element)
    
    bin_mask = inverted_opened_img > 0
    distance=distance_transform_edt(bin_mask)
    
    #create starting points for the watershed
    coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)),labels=inverted_opened_img, min_distance=min_distance_seg_init)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    
    # watershed segmentation
    segmented_image_raw=segmentation.watershed(image=-distance, markers=markers, mask=inverted_opened_img)
    
    #we shuffle the labels since that makes neighboring segments more different in color when plotting
    shuffled_labeled_image = shuffle_labels(segmented_image_raw)
    
    return bin_mask_image, distance, shuffled_labeled_image, cleaned_mask_image
    
def plot_filtered(bin_mask_image, bef_filt_image, aft_filt_image):
    fig, axes = plt.subplots(1, 4, figsize=(15, 8))

    # Panel 1: original mask
    axes[0].imshow(bin_mask_image, cmap='gray')
    axes[0].set_title("Binary mask")

    # Panel 2: filtered segmentation on mask
    axes[1].imshow(bin_mask_image, cmap='gray')  # background mask
    axes[1].imshow(bef_filt_image, cmap='nipy_spectral', alpha=0.6)  # overlay
    axes[1].set_title("Before filtering")


    # Panel 3: overlay filtered_2 on mask
    axes[2].imshow(bin_mask_image, cmap='gray')  # background mask
    axes[2].imshow(aft_filt_image, cmap='nipy_spectral', alpha=0.6)  # overlay
    axes[2].set_title("After filtering")

    # Panel 4: overlay removed on mask
    axes[3].imshow(bin_mask_image, cmap='gray')  # background mask
    axes[3].imshow(bef_filt_image - aft_filt_image, cmap='nipy_spectral', alpha=0.6)  # overlay
    axes[3].set_title("Filtered objects")

    # Remove axes for clarity
    for ax in axes:
        ax.axis('off')

    plt.suptitle(library_id)
    plt.tight_layout()
    plt.show()


def size_shape_filter(seg_input_image, distance_image, min_dist, max_dist, min_area, max_area):
    #filters on object area and max 
    filtered = seg_input_image.copy()
    props = measure.regionprops(seg_input_image, intensity_image=distance_image)
    
    for prop in props:
        label_id = prop.label
        if label_id == 0:
            continue  # skip background

        # compute criteria
        max_dist_this_label = prop.max_intensity   # max distance inside the region
        area = prop.area

        # thresholds (set to what makes sense in your case)
        too_small_dist = max_dist_this_label < min_dist #typically a long, thin object
        too_large_dist = max_dist_this_label > max_dist #less useful, similar to size
        too_small_area = area < min_area
        too_large_area = area > max_area

        if too_small_dist or too_large_dist or too_small_area or too_large_area:
            filtered[seg_input_image == label_id] = 0
    return filtered

def far_from_shore_filter(seg_input_image, bin_tissue_image, min_shore_intensity = 30):
    #removes object that are far from land 
    blurred_image = seg_utils.Preprocessing.apply_gaussian_filter(bin_tissue_image, sigma=40)
    filtered = seg_input_image.copy()
    props = measure.regionprops(seg_input_image, intensity_image=blurred_image)
    
    for prop in props:
        label_id = prop.label
        if label_id == 0:
            continue  # skip background

        # compute criteria
        to_low_smoothed = prop.min_intensity < min_shore_intensity

        if to_low_smoothed:
            filtered[seg_input_image == label_id] = 0
            
    return filtered


def in_the_middle_of_the_ocean_filter_iter(segmented_final, tissue_mask, ring_size=10, min_tissue_overlap=0.2):
    """
    Filters objects in segmented_final based on how much of their perimeter (ring) overlaps tissue.
    Uses a buffered patch around each object to allow full dilation and avoids truncation.
    
    Parameters
    ----------
    segmented_final : 2D ndarray
        Labeled segmented image.
    tissue_mask : 2D ndarray (binary)
        Mask of tissue regions.
    ring_size : int
        Radius of the ring to check around each object.
    min_tissue_overlap : float
        Minimum fraction of ring pixels that must overlap tissue to keep the object.

    Returns
    -------
    filtered : 2D ndarray
        Labeled image with objects removed that are "in the ocean".
    removed_ids : list
        Labels of objects that were removed.
    kept_ids : list
        Labels of objects that were kept.
    """
    filtered = segmented_final.copy()
    removed_ids, kept_ids = [], []
    selem = disk(ring_size)
    nrows, ncols = segmented_final.shape
    tissue_mask_bin = tissue_mask != 0

    for region in regionprops(segmented_final):
        obj_id = region.label
        minr, minc, maxr, maxc = region.bbox

        # --- define buffered patch ---
        r0 = max(minr - ring_size, 0)
        r1 = min(maxr + ring_size, nrows)
        c0 = max(minc - ring_size, 0)
        c1 = min(maxc + ring_size, ncols)

        # Extract patch and binary object mask
        obj_patch = segmented_final[r0:r1, c0:c1] == obj_id
        tissue_patch = tissue_mask_bin[r0:r1, c0:c1]
        seg_patch = segmented_final[r0:r1, c0:c1]

        # Dilate object and compute ring
        ring = binary_dilation(obj_patch, selem) & ~obj_patch

        # Remove pixels belonging to other objects
        ring = ring & (seg_patch == 0)

        # Compute fraction of ring overlapping tissue
        ring_size_nonzero = np.sum(ring)
        if ring_size_nonzero == 0:
            frac_overlap = 0.0
        else:
            frac_overlap = np.sum(tissue_patch[ring]) / ring_size_nonzero
        #print(f"id: {obj_id}, frac_overlap: {frac_overlap}")
        #print(np.sum(tissue_patch[ring]))
        #print(ring_size_nonzero)
        # Decide to keep or remove
        if frac_overlap < min_tissue_overlap:
            filtered[r0:r1, c0:c1][obj_patch] = 0
            removed_ids.append(obj_id)
        else:
            kept_ids.append(obj_id)

    return filtered, removed_ids, kept_ids

def in_the_middle_of_the_ocean_filter(segmented_input, tissue_mask, ring_size=10, min_tissue_overlap=0.2):
    #we iterate several times until nothing more is removed. This is because ocean objects can be in layers,
    #where in the first iteration an ocean object can be saved by another object on its outside.
    keep_going = True
    removed_ids = []
    filtered = segmented_input
    while (keep_going):
        filtered, rem_ids, kept_ids = in_the_middle_of_the_ocean_filter_iter(filtered, tissue_mask, ring_size=ring_size, min_tissue_overlap=min_tissue_overlap)
        removed_ids.append(rem_ids)
        keep_going = len(rem_ids) > 0
        print(f"Removed {len(rem_ids)} objects")
    return filtered, removed_ids, kept_ids
    
def remove_jagged_spindle_objects(segmented_image, ecc_thresh=0.8, compactness_thresh=0.5):
    """
    Remove objects that are elongated and have jagged edges.
    """
    filtered = segmented_image.copy()
    removed_ids = []

    for region in regionprops(segmented_image):
        ecc = region.eccentricity
        if region.perimeter == 0:
            continue  # avoid division by zero
        compactness = 4 * np.pi * region.area / (region.perimeter ** 2)

        if ecc > ecc_thresh and compactness < compactness_thresh:
            # object is elongated and jagged → remove
            filtered[segmented_image == region.label] = 0
            removed_ids.append(region.label)

    return filtered, removed_ids


def evaluate_segmentation(segmented_final, segmented_ground_truth, overlap_threshold = 0.4):
    """
    Fast evaluation against ground truth using Jaccard index (IoU), selecting matches by *max IoU*.

    Returns a dict with:
        - TP, FP, FN: segment IDs (TP/FP from 'segmented_final', FN are GT labels)
        - TP_labels_in_GT: GT label matched to each TP
        - FP_best_GT: best-overlap GT label for each FP (or None)
        - FN_labels_in_GT: GT labels that were missed
        - tp_coverage, fp_coverage, fn_coverage: best IoU values for each list above
    """

    # Flatten
    pred = segmented_final.ravel()
    gt   = segmented_ground_truth.ravel()

    # Unique labels (exclude background=0) and their true pixel counts
    f_labels, f_counts = np.unique(pred[pred > 0], return_counts=True)
    g_labels, g_counts = np.unique(gt[gt > 0],   return_counts=True)
    nf, ng = len(f_labels), len(g_labels)

    # Short-circuit trivial case
    if nf == 0 and ng == 0:
        return {
            "TP": [], "FP": [], "FN": [],
            "TP_labels_in_GT": [], "FP_best_GT": [], "FN_labels_in_GT": [],
            "tp_coverage": [], "fp_coverage": [], "fn_coverage": []
        }

    # Build sparse intersection matrix only over pixels where both labels > 0
    both = (pred > 0) & (gt > 0)
    if np.any(both):
        pred_pos = pred[both]
        gt_pos   = gt[both]
        # Map raw labels -> compressed indices [0..nf-1], [0..ng-1]
        f_idx = np.searchsorted(f_labels, pred_pos)
        g_idx = np.searchsorted(g_labels, gt_pos)
        data  = np.ones_like(f_idx, dtype=np.int64)
        inter = coo_matrix((data, (f_idx, g_idx)), shape=(nf, ng)).tocsr()
    else:
        from scipy.sparse import csr_matrix
        inter = csr_matrix((nf, ng), dtype=np.int64)

    # Prepare outputs
    TP, FP, FN = [], [], []
    TP_labels_in_GT, FP_best_GT, FN_labels_in_GT = [], [], []
    tp_coverage, fp_coverage, fn_coverage = [], [], []

    # Convenience handles for CSR
    indptr, indices, values = inter.indptr, inter.indices, inter.data

    # --- Classify each predicted segment as TP or FP by max IoU over overlapping GTs ---
    matched_gt_labels = set()
    for i in range(nf):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            FP.append(int(f_labels[i]))
            FP_best_GT.append(None)
            fp_coverage.append(0.0)
            continue

        cols = indices[start:end]            # GT indices overlapping this predicted segment
        inter_vals = values[start:end].astype(np.float64)
        unions = f_counts[i] + g_counts[cols] - inter_vals
        ious = inter_vals / unions

        best_k = int(np.argmax(ious))
        j = int(cols[best_k])
        best_iou = float(ious[best_k])

        if best_iou >= overlap_threshold:
            TP.append(int(f_labels[i]))
            TP_labels_in_GT.append(int(g_labels[j]))
            tp_coverage.append(best_iou)
            matched_gt_labels.add(int(g_labels[j]))
        else:
            FP.append(int(f_labels[i]))
            FP_best_GT.append(int(g_labels[j]))
            fp_coverage.append(best_iou)

    # --- Now mark any GT segment not matched as FN ---
    for j, g in enumerate(g_labels):
        if g not in matched_gt_labels:
            FN.append(None)
            FN_labels_in_GT.append(int(g))
            fn_coverage.append(0.0)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TP_labels_in_GT": TP_labels_in_GT,
        "FP_best_GT": FP_best_GT,
        "FN_labels_in_GT": FN_labels_in_GT,
        "tp_coverage": tp_coverage,
        "fp_coverage": fp_coverage,
        "fn_coverage": fn_coverage
    }

def calc_f1(metrics):
    tp = len(metrics['TP'])
    fp = len(metrics['FP'])
    fn = len(metrics['FN'])
    P = tp / (tp + fp) if (tp + fp) else 0.0 #precision
    R = tp / (tp + fn) if (tp + fn) else 0.0 #recall
    F1 = 2*P*R/(P+R) if (P+R) else 0.0
    return P, R, F1

def get_pred_eval_dataframe(gt_ids, old_metrics, new_metrics, train_ids, val_ids, test_ids):
    rows = []
    for gt_id, m_old, m_new in zip(gt_ids, old_metrics, new_metrics):
        P_old, R_old, F1_old = calc_f1(m_old)
        P_new, R_new, F1_new = calc_f1(m_new)
        
        if gt_id in train_ids:
            split = "train"
        elif gt_id in val_ids:
            split = "val"
        elif gt_id in test_ids:
            split = "test"
        else:
            split = "unknown"
        
        rows.append({
            "gt_id": gt_id,
            "split": split,
            "P_old": P_old, "R_old": R_old, "F1_old": F1_old,
            "P_new": P_new, "R_new": R_new, "F1_new": F1_new
        })

    df = pd.DataFrame(rows)
    return df


def plot_overlap_histograms(tp_coverage, fp_coverage, fn_coverage, bins=20):
    """
    Plot separate histograms for TP, FP, and FN side by side.

    tp_coverage, fp_coverage, fn_coverage: lists of overlap fractions (0..1)
    bins: number of bins for histograms
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    axes[0].hist(tp_coverage, bins=bins, color='green', alpha=0.7, range=(0,1))
    axes[0].set_title("True Positives (TP)")
    axes[0].set_xlabel("Best Overlap Fraction")
    axes[0].set_ylabel("Number of Segments")

    axes[1].hist(fp_coverage, bins=bins, color='red', alpha=0.7, range=(0,1))
    axes[1].set_title("False Positives (FP)")
    axes[1].set_xlabel("Best Overlap Fraction")

    axes[2].hist(fn_coverage, bins=bins, color='blue', alpha=0.7, range=(0,1))
    axes[2].set_title("False Negatives (FN)")
    axes[2].set_xlabel("Best Overlap Fraction")

    plt.tight_layout()
    plt.show()

def plot_missed_segments(segmented_final, segmented_ground_truth, tp_labels, overlap_threshold):
    """
    Plot two versions of the ground truth with true positives removed or partially masked.
    
    segmented_final: labeled final segmentation
    segmented_ground_truth: labeled ground truth
    tp_labels: list of labels from segmented_final that were counted as TP
    overlap_threshold: fraction threshold used to define TP
    """
    # Copy ground truth to avoid modifying original
    gt_removed = segmented_ground_truth.copy()
    gt_masked = segmented_ground_truth.copy()
    
    # For each TP segment, find the best overlapping GT segment
    for f in tp_labels:
        mask_f = segmented_final == f
        gt_ids, counts = np.unique(segmented_ground_truth[mask_f], return_counts=True)
        overlaps = {g: c for g, c in zip(gt_ids, counts) if g != 0}
        if not overlaps:
            continue
        # Get the GT segment with largest overlap
        best_gt = max(overlaps, key=overlaps.get)
        frac = overlaps[best_gt] / np.sum(mask_f)
        if frac >= overlap_threshold:
            # 1️⃣ Remove all pixels of that GT segment
            gt_removed[gt_removed == best_gt] = 0
            # 2️⃣ Mask out overlapping pixels only
            overlap_mask = (mask_f) & (segmented_ground_truth == best_gt)
            gt_masked[overlap_mask] = 0

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(gt_removed, cmap='nipy_spectral')
    axes[0].set_title("GT with TP segments removed entirely")
    axes[0].axis('off')
    
    axes[1].imshow(gt_masked, cmap='nipy_spectral')
    axes[1].set_title("GT with overlapping parts of TP masked out")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_false_positives(segmented_final, FP_labels):
    """
    Plot the false positive segments from the predicted segmentation.

    Parameters
    ----------
    segmented_final : np.ndarray
        Labeled predicted segmentation (0 = background)
    FP_labels : list
        List of predicted segment labels that were counted as false positives
    """
    # Create a mask with only the false positives
    fp_mask = np.isin(segmented_final, FP_labels)
    
    # Optional: show each FP segment with a colormap
    # We'll create a temporary image where each FP segment keeps its label for coloring
    fp_image = np.where(fp_mask, segmented_final, 0)

    plt.figure(figsize=(6,6))
    plt.imshow(fp_image, cmap='nipy_spectral')
    plt.title("False Positive Segments")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compute_segmentation_metrics(metrics_dict):
    """
    Compute precision, recall, F1, and mean Jaccard from segmentation results.

    Parameters
    ----------
    metrics_dict : dict
        Output from evaluate_segmentation_jaccard or similar function.
        Must contain:
            - TP, FP, FN : lists of segment IDs
            - tp_coverage : list of Jaccard indices for TPs

    Returns
    -------
    results : dict
        Dictionary with precision, recall, F1, mean_IoU_TP
    """
    TP = metrics_dict["TP"]
    FP = metrics_dict["FP"]
    FN = metrics_dict["FN"]
    tp_coverage = metrics_dict["tp_coverage"]

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0.0
    recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_IoU_TP = np.mean(tp_coverage) if tp_coverage else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "mean_IoU_TP": mean_IoU_TP
    }

def extract_shape_size_features(segmented_image, distance_image):
    """
    Extract features for each segmented object.
    
    Features included:
      - segment_id
      - area (as sqrt(area))
      - eccentricity
      - compactness
      - max_dist (from distance transform, requires distance_image)
      - distance_from_shore (min intensity in smoothed shore mask, requires shore_image)

    Parameters
    ----------
    segmented_image : 2D array
        Labeled segmentation mask.
    distance_image : 2D array, optional
        Distance transform image used to compute max_dist feature.
    shore_image : 2D array, optional
        Smoothed tissue mask (e.g. Gaussian-blurred binary mask of tissue).
    
    Returns
    -------
    pandas.DataFrame
    """
    features = []

    props = regionprops(segmented_image, intensity_image=distance_image)

    for region in props:
        ecc = region.eccentricity
        area = region.area
        if region.perimeter == 0: #div by zero otherwise
            compactness = 0
        else:
            compactness = 4 * np.pi * region.area / (region.perimeter ** 2)

        # distance-based feature
        max_dist = region.max_intensity

        features.append({
            "segment_id": region.label,
            "area": np.sqrt(area),
            "eccentricity": ecc,
            "compactness": compactness,
            "max_dist": max_dist,
        })

    df = pd.DataFrame(features)
    return df


def extract_distance_from_shore(segmented_image, bin_tissue_image, smoothing_sigma = 40):
    """
    Measure how close each object is to tissue/shore.

    Parameters
    ----------
    segmented_image : 2D array
        Labeled segmentation mask.
    bin_tissue_image : 2D array
        Tissue mask (e.g. Gaussian-blurred binary mask of tissue).

    Returns
    -------
    pandas.DataFrame
        Columns: segment_id, distance_from_shore
    """
    features = []

    #removes object that are far from land 
    blurred_image = seg_utils.Preprocessing.apply_gaussian_filter(bin_tissue_image, sigma=smoothing_sigma)
    props = measure.regionprops(segmented_image, intensity_image=blurred_image)
    
    props = regionprops(segmented_image, intensity_image=blurred_image)

    for region in props:
        # min intensity inside blurred shore mask tells how far from "shore" object is
        distance_from_shore = region.min_intensity
        features.append({
            "segment_id": region.label,
            "distance_from_shore": distance_from_shore
        })

    return pd.DataFrame(features)

def compute_ring_features(segmented_image, tissue_mask, ring_size=10):
    """
    Compute features for each object in segmented_image:
        - fraction of ring covered by other objects
        - fraction of ring covered by tissue
    
    Parameters
    ----------
    segmented_image : 2D ndarray
        Labeled segmented image
    tissue_mask : 2D ndarray (binary)
        Binary mask of tissue
    ring_size : int
        Radius of ring for feature calculation
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
        ['segment_id', 'frac_ring_other_objects', 'frac_ring_tissue']
    """
    object_ids = np.unique(segmented_image)
    object_ids = object_ids[object_ids != 0]  # exclude background
    
    selem = disk(ring_size)
    tissue_mask_bin = tissue_mask != 0
    features = []

    nrows, ncols = segmented_image.shape

    for region in regionprops(segmented_image):
        obj_id = region.label
        minr, minc, maxr, maxc = region.bbox

        # buffered patch
        r0 = max(minr - ring_size, 0)
        r1 = min(maxr + ring_size, nrows)
        c0 = max(minc - ring_size, 0)
        c1 = min(maxc + ring_size, ncols)

        obj_patch = segmented_image[r0:r1, c0:c1] == obj_id
        seg_patch = segmented_image[r0:r1, c0:c1]
        tissue_patch = tissue_mask_bin[r0:r1, c0:c1]

        # ring around object
        ring = binary_dilation(obj_patch, selem) & ~obj_patch

        # feature 1: fraction of ring covered by other objects
        ring_other_objects = ring & (seg_patch != 0)
        ring_size_nonzero = np.sum(ring)
        frac_other_objects = np.sum(ring_other_objects) / ring_size_nonzero if ring_size_nonzero > 0 else 0.0

        # feature 2: fraction of ring covered by tissue
        frac_tissue = np.sum(tissue_patch[ring]) / ring_size_nonzero if ring_size_nonzero > 0 else 0.0

        features.append({
            'segment_id': obj_id,
            'frac_ring_other_objects': frac_other_objects,
            'frac_ring_tissue': frac_tissue
        })

    df = pd.DataFrame(features)
    return df


def remove_segments(segmented, remove_ids):
    """
    Remove specific objects (by label) from a segmented image.

    Parameters
    ----------
    segmented : 2D ndarray of int
        Labeled segmented image (0 = background).
    remove_ids : list of int
        Labels of objects to remove.

    Returns
    -------
    filtered : 2D ndarray of int
        New segmented image with selected labels removed (set to 0).
    """
    filtered = segmented.copy()
    if len(remove_ids) == 0:
        return filtered

    mask = np.isin(filtered, remove_ids)
    filtered[mask] = 0
    return filtered

def prefilter_image(preprocessed_seg_image, distance_image):
    '''
    Removes really large and really small objects
    '''
    bef_filtering = shuffled_labeled_image
    prefiltered = size_shape_filter(bef_filtering, distance_image, min_dist=5, max_dist=200, min_area=30, max_area=12000)
    #plot_filtered(bin_mask_image, bef_filtering, prefiltered)
    return prefiltered

def calculate_features(prefiltered_seg_image, distance_image, bin_mask_image):
    sz_feat = extract_shape_size_features(prefiltered_seg_image, distance_image)
    dfs_feat = extract_distance_from_shore(prefiltered_seg_image, bin_mask_image)
    itmoto_feat = compute_ring_features(prefiltered_seg_image, bin_mask_image, ring_size=10)
    # Merge the three feature dataframes on 'obj_id'
    merged_feat = (
        sz_feat
        .merge(dfs_feat, on="segment_id", how="outer")
        .merge(itmoto_feat, on="segment_id", how="outer")
    )
    #very small objects get NaN values for these features, set those to zero
    cols_to_fill = ['area', 'eccentricity', 'compactness', 'max_dist']
    merged_feat[cols_to_fill] = merged_feat[cols_to_fill].fillna(0)

    return merged_feat


def assign_gt_labels(segmented_final, segmented_ground_truth, overlap_threshold=0.5):
    """
    Assign each predicted object in segmented_final the GT label with which it has
    the highest IoU (>= overlap_threshold). If no GT passes threshold, assign 0.

    Returns:
        mapping: dict {pred_label -> assigned_gt_label}
                 (background excluded, unmatched preds map to 0)
    """

    pred = segmented_final.ravel()
    gt   = segmented_ground_truth.ravel()

    # Unique nonzero labels
    f_labels, f_counts = np.unique(pred[pred > 0], return_counts=True)
    g_labels, g_counts = np.unique(gt[gt > 0],   return_counts=True)
    nf, ng = len(f_labels), len(g_labels)

    # Empty case
    if nf == 0 or ng == 0:
        return {int(f): 0 for f in f_labels}

    # Build sparse intersection matrix only where pred>0 and gt>0
    both = (pred > 0) & (gt > 0)
    if np.any(both):
        f_idx = np.searchsorted(f_labels, pred[both])
        g_idx = np.searchsorted(g_labels, gt[both])
        data  = np.ones_like(f_idx, dtype=np.int64)
        inter = coo_matrix((data, (f_idx, g_idx)), shape=(nf, ng)).tocsr()
    else:
        inter = csr_matrix((nf, ng), dtype=np.int64)

    mapping = {}
    indptr, indices, values = inter.indptr, inter.indices, inter.data

    for i in range(nf):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            mapping[int(f_labels[i])] = 0
            continue

        cols = indices[start:end]
        inter_vals = values[start:end].astype(np.float64)
        unions = f_counts[i] + g_counts[cols] - inter_vals
        ious = inter_vals / unions

        best_k = int(np.argmax(ious))
        j = int(cols[best_k])
        best_iou = float(ious[best_k])

        if best_iou >= overlap_threshold:
            mapping[int(f_labels[i])] = int(g_labels[j])
        else:
            mapping[int(f_labels[i])] = 0

    return mapping

def preprocess_and_extract(img_id, input_img, gt_img, iou_threshold=0.5, 
                           min_distance_seg_init=30, min_island_area = 0):
    """
    Preprocess input image, segment, extract features, and assign ground truth labels.
    """
    # Preprocess and segment
    bin_mask, distance, segmented, cleaned_mask_image = preprocess_input_image(input_img, 
                                                               min_distance_seg_init = min_distance_seg_init,
                                                               min_island_area = min_island_area)

    # Features
    features_df = calculate_features(segmented, distance, bin_mask)
    #features_df = features_df.set_index("segment_id", drop=False)

    # Assign GT per object via IoU matching
    if gt_img is None:
        features_df["ground_truth"] = np.nan
    else:
        gt_map = assign_gt_labels(segmented, gt_img, iou_threshold)
        #replace NA (objects with no overlap with ground truth) with zeros and convert to int (NA forces float)
        features_df["ground_truth"] = features_df["segment_id"].map(gt_map).fillna(0).astype(float)
    features_df["image_id"] = img_id

    return features_df, bin_mask, distance, segmented, cleaned_mask_image


def prepare_datasets(all_ids, gt_ids, input_images, ground_truth_images, min_distance_seg_init = 30,
                    min_island_area = 0):
    """
    Preprocess all images and extract features + labels.
    The returned ground_truth number is the best overlapping label in the gt if above the 
    coverage threshold, 0 if matched but below the threshold, and NaN if no overlap. So, 
    we treat NaN and 0 as False. Note that this classifier we are building assumes the overlaps
    of real objects are good - the job here is to decide which objects are good objects, not to decide if
    the watershed did a good job
    """
    # 1. Collect object-level features for all images
    all_dfs = []
    all_bin_mask = []
    all_distance = []
    all_segmented = []
    all_cleaned_masks = []
    for img_id, input_img in zip(all_ids, input_images):
        #look up the ground truth image if it exists
        gt_ind = [i for i,x in enumerate(gt_ids) if x == img_id]
        if len(gt_ind) == 1:
            gt_img = ground_truth_images[gt_ind[0]]
        else:
            gt_img = None
        print(f"Processing {img_id}")
        df, bin_mask, distance, segmented, cleaned_mask_image = preprocess_and_extract(img_id, input_img, gt_img, min_distance_seg_init=min_distance_seg_init, min_island_area=min_island_area)
        all_dfs.append(df)
        all_bin_mask.append(bin_mask)
        all_distance.append(distance)
        all_segmented.append(segmented)
        all_cleaned_masks.append(cleaned_mask_image)

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    return full_df, all_bin_mask, all_distance, all_segmented, all_cleaned_masks

def plot_segmented_image(full_df, all_segmented, gt_ids, index):
    """
    Plot a segmented image with two panels: ground_truth != 0 and ground_truth == 0.

    Parameters
    ----------
    full_df : pd.DataFrame
        Must contain 'image_id', 'segment_id', 'ground_truth'.
    all_segmented : list of np.array
        Segmented images (2D arrays of segment labels).
    gt_ids : list
        Image IDs corresponding to all_segmented (same order).
    index : int
        Index into all_segmented / gt_ids.
    """
    seg_image = all_segmented[index]
    image_id = gt_ids[index]

    # Get all segments for this image
    image_df = full_df[full_df['image_id'] == image_id]

    # Create masks for segments
    gt_segments = image_df[image_df['ground_truth'] != 0]['segment_id'].values
    non_gt_segments = image_df[image_df['ground_truth'] == 0]['segment_id'].values

    # Boolean masks for plotting
    mask_gt = np.isin(seg_image, gt_segments)
    mask_non_gt = np.isin(seg_image, non_gt_segments)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mask_gt, cmap='Reds')
    axes[0].set_title(f"{image_id}: ground_truth != 0")
    axes[0].axis('off')

    axes[1].imshow(mask_non_gt, cmap='Blues')
    axes[1].set_title(f"{image_id}: ground_truth == 0")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_one_segment(segmented_image, segment_label):
    """
    Plot only the specified segment in a labeled image.

    Parameters
    ----------
    segmented_image : 2D array
        Labeled segmentation mask.
    segment_label : int
        The label of the segment to keep.

    """
    new_image = np.zeros_like(segmented_image)
    new_image[segmented_image == segment_label] = segment_label
    bin_image =  new_image > 0

    # Plot the result
    plt.figure(figsize=(6,6))
    plt.imshow(bin_image, cmap='nipy_spectral')
    plt.title('Single Segment')
    plt.axis('off')
    plt.show()

def get_training_input_from_df(df):
    #scale them
    X = df.drop(columns=['segment_id','ground_truth','image_id']).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    return X

def get_y_from_df(df):
    y = (df['ground_truth'] > 0).to_numpy().astype(np.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return y

class AdipoNN(nn.Module):
    def __init__(self, input_dim):
        super(AdipoNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
            )
        
    def forward(self, x):
        return self.net(x)


def save_model(filename, model, train_ids, val_ids, test_ids):
    input_dim = model.net[0].in_features
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }, filename)

def load_model(filename):
    '''used when training and when loading files'''
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    model = AdipoNN(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["train_ids"], checkpoint["val_ids"], checkpoint["test_ids"]

def set_seed(seed=42):
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # PyTorch GPU
        torch.cuda.manual_seed_all(seed)       # if using multi-GPU
        torch.backends.cudnn.deterministic = True  # make CUDA deterministic
        torch.backends.cudnn.benchmark = False     # disable auto-optimization

def train_model(full_df, n_epochs = 1000, seed = 42, val_frac = 0.2, test_frac = 0.2):
    set_seed(seed)
    image_ids = full_df['image_id'].unique()
    train_ids, test_ids = train_test_split(image_ids, test_size=test_frac, random_state=42) #don't use seed here, better to always get the same
    train_ids, val_ids = train_test_split(train_ids, test_size=val_frac, random_state=42)  # split
    print(f"Training: {len(train_ids)}, validation: {len(val_ids)}, test: {len(test_ids)}")


    train_mask = full_df['image_id'].isin(train_ids)
    val_mask   = full_df['image_id'].isin(val_ids)
    test_mask  = full_df['image_id'].isin(test_ids)


    X_train = get_training_input_from_df(full_df.loc[train_mask])
    y_train = get_y_from_df(full_df.loc[train_mask])

    X_val   = get_training_input_from_df(full_df.loc[val_mask])
    y_val   = get_y_from_df(full_df.loc[val_mask])

    X_test  = get_training_input_from_df(full_df.loc[test_mask])
    y_test  = get_y_from_df(full_df.loc[test_mask])


    #def worker_init_fn(worker_id):
        # Each worker has a different RNG, we seed it based on global seed + worker id
    #    seed = torch.initial_seed() % 2**32
    #    np.random.seed(seed)
    #    random.seed(seed)


    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = AdipoNN(X_train.shape[1])
    
    # Loss and optimizer
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss:.4f}")
        
    return model, train_ids, val_ids, test_ids
        
def evaluate_model(model, full_df, test_ids):
    # Make sure your model is in evaluation mode
    model.eval()
    test_mask  = full_df['image_id'].isin(test_ids)
    
    X_test  = get_training_input_from_df(full_df.loc[test_mask])
    y_test  = get_y_from_df(full_df.loc[test_mask])

    # Disable gradient computation for evaluation
    with torch.no_grad():
        y_pred_prob = model(X_test)  # outputs between 0 and 1
        y_pred = (y_pred_prob > 0.5).int()  # convert to 0 or 1

    # Convert tensors back to numpy for sklearn metrics
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()

    # Classification report
    print(classification_report(y_test_np, y_pred_np))
    
def prediction_plot(bin_mask_image, segmented_unfiltered, pred_image, gt_image, filename = None, show_plot = True):
    if gt_image is None:    
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    # Panel 1: original mask
    axes[0].imshow(bin_mask_image, cmap='gray')
    axes[0].set_title("Binary mask")

    axes[1].imshow(bin_mask_image, cmap='gray')  # background mask
    axes[1].imshow(segmented_unfiltered, cmap=get_seg_colormap(), alpha=0.6)  # overlay
    axes[1].set_title("Unfiltered")
    
    # Panel 3: filtered segmentation on mask
    axes[2].imshow(bin_mask_image, cmap='gray')  # background mask
    axes[2].imshow(pred_image, cmap=get_seg_colormap(), alpha=0.6)  # overlay
    axes[2].set_title("Predicted")

    # Panel 4: overlay filtered_2 on mask
    if not gt_image is None:    
        axes[3].imshow(bin_mask_image, cmap='gray')  # background mask
        axes[3].imshow(gt_image, cmap=get_seg_colormap(), alpha=0.6)  # overlay
        axes[3].set_title("Ground truth")

    # Remove axes for clarity
    for ax in axes:
        ax.axis('off')

    plt.suptitle("Prediction plot")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

def predict_and_clean_image(model, samp_id, samp_ind, full_df, unfiltered_seg):
    df_samp = full_df.loc[full_df['image_id'] == samp_id]
    X_samp = get_training_input_from_df(df_samp)
    segmented_sample = unfiltered_seg
    model.eval()
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(model(X_samp)).numpy().flatten()
        y_pred = (y_pred_prob > 0.5).astype(bool)

    segment_ids = df_samp['segment_id'].values
    segment_ids_to_rem = segment_ids[~y_pred]

    cleaned_image = np.copy(segmented_sample)
    for seg_id in segment_ids_to_rem:
        cleaned_image[cleaned_image == seg_id] = 0
    return cleaned_image

def predict_and_clean_images(model, full_df, all_ids, all_unfiltered_seg, 
                             gt_ids, gt_images, all_bin_mask, path, plot_path):
    '''Paths are expected to end with /'''
    os.makedirs(path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
        
    for samp_ind, samp_id in enumerate(all_ids):
        print(f"Processing {samp_ind+1} of {len(all_ids)}")
        cleaned_image = predict_and_clean_image(model, samp_id, samp_ind, full_df, all_unfiltered_seg[samp_ind])
        
        #look up the ground truth image if it exists
        gt_ind = [i for i,x in enumerate(gt_ids) if x == samp_id]
        if len(gt_ind) == 1:
            gt_img = gt_images[gt_ind[0]]
        else:
            gt_img = None
        img = Image.fromarray(cleaned_image.astype(np.uint16))
        img.save(path + samp_id + " pred_adipocytes.png")
        
        #also make a plot
        prediction_plot(all_bin_mask[samp_ind], all_unfiltered_seg[samp_ind], cleaned_image, 
                        gt_img, filename = plot_path + samp_id + " prediction_plot.png", show_plot = False)

