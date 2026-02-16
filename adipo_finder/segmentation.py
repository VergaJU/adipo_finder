import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage import feature, measure, morphology, segmentation


class Segmentation:

    @staticmethod
    def find_local_maxima(
        image: np.ndarray, min_distance: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find local maxima in the distance transform.
        Add additional negative markers from the inverted image to better identify
        merged adipocytes.

        parameters:
        image: np.ndarray, input image
        min_distance: int, minimum distance separating peaks

        return:
        tuple[np.ndarray, np.ndarray]: distance transform, markers
        """
        distance = distance_transform_edt(image > 0)
        coords = feature.peak_local_max(
            distance, footprint=np.ones((3, 3)), labels=image, min_distance=min_distance
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = label(mask)

        return distance, markers

    @staticmethod
    def apply_watershed_segmentation(
        image: np.ndarray,
        markers: np.ndarray,
        distance: np.ndarray,
        window: int = 20,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply watershed segmentation to the image. Perform opening to disconnect touching adipocytes

        parameters:
        image: np.ndarray, input image
        markers: np.ndarray, markers for the watershed segmentation
        distance: np.ndarray, distance transform of the image
        window: int, window size for the opening operation

        return:
        np.ndarray, watershed segmentation
        """
        # watershed segmentation
        segmented_image = segmentation.watershed(
            image=-distance, markers=markers, mask=image
        )
        # perform opening
        structuring_element = morphology.disk(window)
        # structuring_element = np.ones((window, window))
        segmented_image = morphology.opening(
            segmented_image, footprint=structuring_element
        )
        return segmented_image

    @staticmethod
    def filter_objects_by_size(
        label_image: np.ndarray, min_size: int = 400, max_size: int = None, **kwargs
    ) -> np.ndarray:
        """
        Filter objects by size.

        parameters:
        label_image: np.ndarray, input segmentation
        min_size: int, minimum size of the objects
        max_size: int, maximum size of the objects

        return:
        np.ndarray, filtered segmentation
        """
        ## remove small objects
        small_removed = (
            morphology.remove_small_objects(label_image.astype(bool), min_size).astype(
                np.uint8
            )
            * 255
        )
        ## if max_size is specified, remove large objects
        if max_size is not None:
            mid_removed = (
                morphology.remove_small_objects(
                    small_removed.astype(bool), max_size
                ).astype(np.uint8)
                * 255
            )
            large_removed = small_removed - mid_removed
            return large_removed.astype(np.uint8).astype(np.uint8) * 255
        else:
            return small_removed

    # function who takes the image and pixels to increase the size of the adipocytes
    @staticmethod
    def expand_adipocytes(image: np.ndarray, pixels: int = 5, **kwargs) -> np.ndarray:
        """
        Expand the size of the adipocytes. Expand each label separated to don't merge them

        parameters:
        image: np.ndarray, input image
        pixels: int, number of pixels to increase the size of the adipocytes

        return:
        np.ndarray, expanded image
        """
        labeled_image = measure.label(image)
        ## create a structuring element
        selem = morphology.disk(pixels)
        # Initialize an empty array for the dilated image
        dilated_image = np.zeros_like(labeled_image)
        # Get the unique labels (ignoring background label 0)
        labels = np.unique(labeled_image)
        labels = labels[labels != 0]
        # Dilate each labeled object separately
        for label in labels:
            # Create a binary mask for the current label
            binary_mask = labeled_image == label

            # Dilate the binary mask
            dilated_mask = binary_dilation(binary_mask, structure=selem)

            # Add the dilated mask to the dilated_image, preserving the label
            dilated_image[dilated_mask] = label

        dilated_image = segmentation.clear_border(dilated_image)
        return dilated_image

    @classmethod
    def run_segmentation(cls, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run the entire segmentation pipeline.

        parameters:
        image: np.ndarray, input image

        return:
        np.ndarray, filtered segmentation
        """
        # distance = cls.compute_distance_transform(image)
        distance, markers = cls.find_local_maxima(image)
        # set kwargs for the function
        kwargs.get("window", 20)
        kwargs.get("min_size", 400)
        kwargs.get("max_size", None)
        kwargs.get("pixels", 5)
        # segment and perform opening
        segmented_image = cls.apply_watershed_segmentation(
            image, markers, distance, **kwargs
        )
        # perform filtering
        filtered_image = cls.filter_objects_by_size(segmented_image, **kwargs)
        # expand the adipocytes
        expanded_image = cls.expand_adipocytes(filtered_image, **kwargs)

        return expanded_image

    @staticmethod
    def size_shape_filter(
        seg_input_image: np.ndarray,
        distance_image: np.ndarray,
        min_dist: float,
        max_dist: float,
        min_area: int,
        max_area: int,
    ) -> np.ndarray:
        """
        Filter objects by size and shape (max distance).
        """
        filtered = seg_input_image.copy()
        props = measure.regionprops(seg_input_image, intensity_image=distance_image)

        for prop in props:
            label_id = prop.label
            if label_id == 0:
                continue  # skip background

            # compute criteria
            max_dist_this_label = prop.max_intensity  # max distance inside the region
            area = prop.area

            # thresholds (set to what makes sense in your case)
            too_small_dist = (
                max_dist_this_label < min_dist
            )  # typically a long, thin object
            too_large_dist = (
                max_dist_this_label > max_dist
            )  # less useful, similar to size
            too_small_area = area < min_area
            too_large_area = area > max_area

            if too_small_dist or too_large_dist or too_small_area or too_large_area:
                filtered[seg_input_image == label_id] = 0
        return filtered

    @staticmethod
    def far_from_shore_filter(
        seg_input_image: np.ndarray,
        bin_tissue_image: np.ndarray,
        min_shore_intensity: int = 30,
    ) -> np.ndarray:
        """
        Removes object that are far from land (tissue).
        """
        from .utils import Preprocessing

        blurred_image = Preprocessing.apply_gaussian_filter(bin_tissue_image, sigma=40)
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

    @staticmethod
    def in_the_middle_of_the_ocean_filter_iter(
        segmented_final: np.ndarray,
        tissue_mask: np.ndarray,
        ring_size: int = 10,
        min_tissue_overlap: float = 0.2,
    ) -> tuple[np.ndarray, list, list]:
        """
        Filters objects in segmented_final based on how much of their perimeter (ring) overlaps tissue.
        Uses a buffered patch around each object to allow full dilation and avoids truncation.
        """
        filtered = segmented_final.copy()
        removed_ids, kept_ids = [], []
        selem = morphology.disk(ring_size)
        nrows, ncols = segmented_final.shape
        tissue_mask_bin = tissue_mask != 0

        for region in measure.regionprops(segmented_final):
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

            # Decide to keep or remove
            if frac_overlap < min_tissue_overlap:
                filtered[r0:r1, c0:c1][obj_patch] = 0
                removed_ids.append(obj_id)
            else:
                kept_ids.append(obj_id)

        return filtered, removed_ids, kept_ids

    @classmethod
    def in_the_middle_of_the_ocean_filter(
        cls,
        segmented_input: np.ndarray,
        tissue_mask: np.ndarray,
        ring_size: int = 10,
        min_tissue_overlap: float = 0.2,
    ) -> tuple[np.ndarray, list, list]:
        """
        Iteratively removes objects not overlapping tissue.
        """
        # we iterate several times until nothing more is removed. This is because ocean objects can be in layers,
        # where in the first iteration an ocean object can be saved by another object on its outside.
        keep_going = True
        removed_ids = []
        filtered = segmented_input
        kept_ids = (
            []
        )  # Initialize kept_ids to be safe, though loop should run at least once logic might vary

        while keep_going:
            filtered, rem_ids, kept_ids = cls.in_the_middle_of_the_ocean_filter_iter(
                filtered,
                tissue_mask,
                ring_size=ring_size,
                min_tissue_overlap=min_tissue_overlap,
            )
            if rem_ids:
                removed_ids.append(rem_ids)
            keep_going = len(rem_ids) > 0
            if keep_going:
                print(f"Removed {len(rem_ids)} objects")
        return filtered, removed_ids, kept_ids

    @staticmethod
    def remove_jagged_spindle_objects(
        segmented_image: np.ndarray,
        ecc_thresh: float = 0.8,
        compactness_thresh: float = 0.5,
    ) -> tuple[np.ndarray, list]:
        """
        Remove objects that are elongated and have jagged edges.
        """
        filtered = segmented_image.copy()
        removed_ids = []

        for region in measure.regionprops(segmented_image):
            ecc = region.eccentricity
            if region.perimeter == 0:
                continue  # avoid division by zero
            compactness = 4 * np.pi * region.area / (region.perimeter**2)

            if ecc > ecc_thresh and compactness < compactness_thresh:
                # object is elongated and jagged → remove
                filtered[segmented_image == region.label] = 0
                removed_ids.append(region.label)

        return filtered, removed_ids

    @classmethod
    def preprocess_input_image(
        cls,
        seg_input_img: np.ndarray,
        min_distance_seg_init: int = 30,
        min_island_area: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess input image, segment it, and shuffle labels.
        """
        from .utils import Preprocessing, shuffle_labels

        bin_mask_image = Preprocessing.segmentation_to_binary(seg_input_img)
        # optionally clean up the image a bit - this will give larger segments in the ocean, but may
        # also make it harder to capture adipocytes on the shores of the continents
        cleaned_mask_image = bin_mask_image.copy()
        if min_island_area > 0:
            labeled_image = measure.label(
                bin_mask_image, connectivity=2
            )  # 2 means 8-connectivity, so also connected via corners
            props = measure.regionprops(labeled_image, intensity_image=bin_mask_image)

            for prop in props:
                label_id = prop.label
                if label_id == 0:
                    continue  # skip background

                if prop.area < min_island_area:
                    cleaned_mask_image[labeled_image == label_id] = 0

        inverted_image = Preprocessing.invert_image(cleaned_mask_image)
        # perform opening
        structuring_element = morphology.disk(4)
        inverted_opened_img = morphology.opening(
            inverted_image, footprint=structuring_element
        )

        bin_mask = inverted_opened_img > 0
        distance = distance_transform_edt(bin_mask)

        # create starting points for the watershed
        distance, markers = cls.find_local_maxima(
            inverted_opened_img, min_distance=min_distance_seg_init
        )

        # watershed segmentation
        segmented_image_raw = segmentation.watershed(
            image=-distance, markers=markers, mask=inverted_opened_img
        )

        # we shuffle the labels since that makes neighboring segments more different in color when plotting
        shuffled_labeled_image = shuffle_labels(segmented_image_raw)

        return bin_mask_image, distance, shuffled_labeled_image, cleaned_mask_image

    @classmethod
    def prefilter_image(
        cls, preprocessed_seg_image: np.ndarray, distance_image: np.ndarray
    ) -> np.ndarray:
        """
        Removes really large and really small objects
        """
        bef_filtering = preprocessed_seg_image
        prefiltered = cls.size_shape_filter(
            bef_filtering,
            distance_image,
            min_dist=5,
            max_dist=200,
            min_area=30,
            max_area=12000,
        )
        return prefiltered


def main():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=50000,
        centers=10,
        n_features=2,
        center_box=(0.0, 1000.0),
        cluster_std=10,
        random_state=0,
    )
    X = X.astype(int)
    image = np.zeros(
        (1000, 1000), dtype=int
    )  # Create, the result array; initialize with 0
    image[X[:, 0], X[:, 1]] = 255
    image = image.astype(np.uint8)  # Use ar as a source of indices, to assign 1
    # image = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    distance, markers = Segmentation.find_local_maxima(image)
    segmented_image = Segmentation.apply_watershed_segmentation(
        image, markers, distance, window=2
    )
    filtered_image = Segmentation.filter_objects_by_size(segmented_image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(segmented_image, cmap="gray")
    axes[1].set_title("Segmented Image")
    axes[2].imshow(filtered_image, cmap="gray")
    axes[2].set_title("Masked Image")
    for ax in axes:
        ax.axis("off")

    plt.savefig("plot.png")  # Specify the path where you want to save the plot
    plt.show()


if __name__ == "__main__":
    main()
