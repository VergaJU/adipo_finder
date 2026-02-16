import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


class Plotting:
    @staticmethod
    def plot_3_channel_image(
        image: np.ndarray, invert_image: np.ndarray, segmented_image: np.ndarray
    ) -> None:
        """
        Create 3 channel image and plot it.

        parameters:
        image: np.ndarray, input image
        invert_image: np.ndarray, inverted image
        segmented_image: np.ndarray, segmented image

        return:
        None
        """
        # Stack images along the last axis to create a 3-channel image
        three_channel_image = np.stack((image, invert_image, segmented_image), axis=-1)

        # Display the 3-channel image using matplotlib
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes.imshow(three_channel_image)
        axes.set_title("3-Channel Image (binary,inverted and segmented)")
        axes.axis("off")

        plt.show()

    @staticmethod
    def plot_centroids(
        image: np.ndarray, size: float = 10.0, figsize: tuple = (10, 10)
    ) -> None:
        """
        Plot the centroids on the image.

        parameters:
        image: np.ndarray, input image

        return:
        None
        """

        labeled_img = measure.label(image)
        regions = measure.regionprops(labeled_img)
        # compute centroids
        centroids = np.array([region.centroid for region in regions])

        # plot labeled image

        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes.imshow(image, cmap="gray")
        axes.scatter(centroids[:, 1], centroids[:, 0], c="r", s=size, marker="+")
        axes.set_title("Centroids")
        axes.axis("off")

        plt.show()


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
    axes[0].imshow(bin_mask_image, cmap="gray")
    axes[0].set_title("Binary mask")

    # Panel 2: unfiltered segmentation on mask
    # Make truncated colormap
    colmap = get_seg_colormap()

    # Convert to array of RGBA colors
    colors = colmap(np.arange(colmap.N))  # shape (n, 4)

    # Prepend black for background
    colors = np.vstack(([0, 0, 0, 1], colors))
    colmap_with_black = mcolors.ListedColormap(colors)

    # TODO: This function seemed incomplete in source, finishing logic based on pattern
    axes[1].imshow(segmented_unfiltered, cmap=colmap_with_black, alpha=0.6)
    axes[1].set_title("Unfiltered")

    axes[2].imshow(pred_image, cmap=colmap_with_black, alpha=0.6)
    axes[2].set_title("Predicted")

    if gt_image is not None:
        axes[3].imshow(gt_image, cmap=colmap_with_black, alpha=0.6)
        axes[3].set_title("Ground Truth")
    else:
        axes[3].axis("off")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_filtered(bin_mask_image, bef_filt_image, aft_filt_image, library_id=""):
    fig, axes = plt.subplots(1, 4, figsize=(15, 8))

    # Panel 1: original mask
    axes[0].imshow(bin_mask_image, cmap="gray")
    axes[0].set_title("Binary mask")

    # Panel 2: filtered segmentation on mask
    axes[1].imshow(bin_mask_image, cmap="gray")  # background mask
    axes[1].imshow(bef_filt_image, cmap="nipy_spectral", alpha=0.6)  # overlay
    axes[1].set_title("Before filtering")

    # Panel 3: overlay filtered_2 on mask
    axes[2].imshow(bin_mask_image, cmap="gray")  # background mask
    axes[2].imshow(aft_filt_image, cmap="nipy_spectral", alpha=0.6)  # overlay
    axes[2].set_title("After filtering")

    # Panel 4: overlay removed on mask
    axes[3].imshow(bin_mask_image, cmap="gray")  # background mask
    axes[3].imshow(
        bef_filt_image - aft_filt_image, cmap="nipy_spectral", alpha=0.6
    )  # overlay
    axes[3].set_title("Filtered objects")

    # Remove axes for clarity
    for ax in axes:
        ax.axis("off")

    plt.suptitle(library_id)
    plt.tight_layout()
    plt.show()


def plot_overlap_histograms(tp_coverage, fp_coverage, fn_coverage, bins=20):
    """
    Plot separate histograms for TP, FP, and FN side by side.

    tp_coverage, fp_coverage, fn_coverage: lists of overlap fractions (0..1)
    bins: number of bins for histograms
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    axes[0].hist(tp_coverage, bins=bins, color="green", alpha=0.7, range=(0, 1))
    axes[0].set_title("True Positives (TP)")
    axes[0].set_xlabel("Best Overlap Fraction")
    axes[0].set_ylabel("Number of Segments")

    axes[1].hist(fp_coverage, bins=bins, color="red", alpha=0.7, range=(0, 1))
    axes[1].set_title("False Positives (FP)")
    axes[1].set_xlabel("Best Overlap Fraction")

    axes[2].hist(fn_coverage, bins=bins, color="blue", alpha=0.7, range=(0, 1))
    axes[2].set_title("False Negatives (FN)")
    axes[2].set_xlabel("Best Overlap Fraction")

    plt.tight_layout()
    plt.show()


def plot_missed_segments(
    segmented_final, segmented_ground_truth, tp_labels, overlap_threshold
):
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

    axes[0].imshow(gt_removed, cmap="nipy_spectral")
    axes[0].set_title("GT with TP segments removed entirely")
    axes[0].axis("off")

    axes[1].imshow(gt_masked, cmap="nipy_spectral")
    axes[1].set_title("GT with overlapping parts of TP masked out")
    axes[1].axis("off")

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

    plt.figure(figsize=(6, 6))
    plt.imshow(fp_image, cmap="nipy_spectral")
    plt.title("False Positive Segments")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


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
    image_df = full_df[full_df["image_id"] == image_id]

    # Create masks for segments
    gt_segments = image_df[image_df["ground_truth"] != 0]["segment_id"].values
    non_gt_segments = image_df[image_df["ground_truth"] == 0]["segment_id"].values

    # Boolean masks for plotting
    mask_gt = np.isin(seg_image, gt_segments)
    mask_non_gt = np.isin(seg_image, non_gt_segments)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mask_gt, cmap="Reds")
    axes[0].set_title(f"{image_id}: ground_truth != 0")
    axes[0].axis("off")

    axes[1].imshow(mask_non_gt, cmap="Blues")
    axes[1].set_title(f"{image_id}: ground_truth == 0")
    axes[1].axis("off")

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
    bin_image = new_image > 0

    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.imshow(bin_image, cmap="nipy_spectral")
    plt.title("Single Segment")
    plt.axis("off")
    plt.show()


def prediction_plot(
    bin_mask_image,
    segmented_unfiltered,
    pred_image,
    gt_image,
    filename=None,
    show_plot=True,
):
    if gt_image is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    # Panel 1: original mask
    axes[0].imshow(bin_mask_image, cmap="gray")
    axes[0].set_title("Binary mask")

    axes[1].imshow(bin_mask_image, cmap="gray")  # background mask
    axes[1].imshow(segmented_unfiltered, cmap=get_seg_colormap(), alpha=0.6)  # overlay
    axes[1].set_title("Unfiltered")

    # Panel 3: filtered segmentation on mask
    axes[2].imshow(bin_mask_image, cmap="gray")  # background mask
    axes[2].imshow(pred_image, cmap=get_seg_colormap(), alpha=0.6)  # overlay
    axes[2].set_title("Predicted")

    # Panel 4: overlay filtered_2 on mask
    if not gt_image is None:
        axes[3].imshow(bin_mask_image, cmap="gray")  # background mask
        axes[3].imshow(gt_image, cmap=get_seg_colormap(), alpha=0.6)  # overlay
        axes[3].set_title("Ground truth")

    # Remove axes for clarity
    for ax in axes:
        ax.axis("off")

    plt.suptitle("Prediction plot")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()
