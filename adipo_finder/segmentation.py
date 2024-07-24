import numpy as np 
from matplotlib import pyplot as plt 
from skimage import measure, morphology, segmentation, feature
from scipy.ndimage import distance_transform_edt, label, binary_dilation
from sklearn.datasets import make_blobs

class Segmentation:

    @staticmethod
    def find_local_maxima(image: np.ndarray) -> np.ndarray:
        """
        Find local maxima in the distance transform. 
        Add additional negative markers from the inverted image to better identify
        merged adipocytes.
        
        parameters:
        image: np.ndarray, input image
        distance: np.ndarray, distance transform of the image

        return:
        np.ndarray, local maxima
        """
        distance=distance_transform_edt(image>0)
        coords = feature.peak_local_max(distance, footprint=np.ones((3, 3)),labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = label(mask)
        
        return distance,markers


    @staticmethod
    def apply_watershed_segmentation(image: np.ndarray, markers: np.ndarray, distance: np.ndarray, window: int = 20,**kwargs) -> np.ndarray:
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
        segmented_image=segmentation.watershed(image=-distance, markers=markers, mask=image)
        # perform opening
        segmented_image = morphology.opening(segmented_image, footprint=np.ones((window,window)))
        return segmented_image
    

    @staticmethod
    def filter_objects_by_size(label_image: np.ndarray, min_size: int = 400, max_size: int = None, **kwargs) -> np.ndarray:
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
        small_removed = morphology.remove_small_objects(label_image.astype(bool), min_size).astype(np.uint8) * 255
        ## if max_size is specified, remove large objects
        if max_size is not None:
            mid_removed = morphology.remove_small_objects(small_removed.astype(bool), max_size).astype(np.uint8) * 255
            large_removed = small_removed - mid_removed
            return large_removed.astype(np.uint8).astype(np.uint8) * 255
        else:
            return small_removed
            
    
    # function who takes the image and pixels to increase the size of the adipocytes
    @staticmethod
    def expand_adipocytes(image: np.ndarray, pixels: int = 5,**kwargs) -> np.ndarray:
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
            binary_mask = (labeled_image == label)
            
            # Dilate the binary mask
            dilated_mask = binary_dilation(binary_mask, structure=selem)
            
            # Add the dilated mask to the dilated_image, preserving the label
            dilated_image[dilated_mask] = label

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
        distance,markers = cls.find_local_maxima(image)
        # set kwargs for the function
        window = kwargs.get('window', 20)
        min_size = kwargs.get('min_size', 400)
        max_size = kwargs.get('max_size', None)
        pixels = kwargs.get('pixels', 5)
        # segment and perform opening
        segmented_image = cls.apply_watershed_segmentation(image, markers, distance, **kwargs)
        # perform filtering
        filtered_image = cls.filter_objects_by_size(segmented_image, **kwargs)
        # expand the adipocytes
        expanded_image = cls.expand_adipocytes(filtered_image, **kwargs)
        return expanded_image





def main():
    from sklearn.datasets import make_blobs
    X,_=make_blobs(n_samples=50000, centers=10, n_features=2,center_box=(0.0, 1000.0),cluster_std=10, random_state=0)
    X=X.astype(int)
    image = np.zeros((1000,1000), dtype=int)  # Create, the result array; initialize with 0
    image[X[:,0], X[:,1]] = 255
    image=image.astype(np.uint8)# Use ar as a source of indices, to assign 1
    # image = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    distance,markers =  Segmentation.find_local_maxima(image)
    segmented_image = Segmentation.apply_watershed_segmentation(image, markers, distance,window=2)
    filtered_image = Segmentation.filter_objects_by_size(segmented_image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title('Segmented Image')
    axes[2].imshow(filtered_image, cmap='gray')
    axes[2].set_title('Masked Image')
    for ax in axes:
        ax.axis('off')

    plt.savefig('plot.png')  # Specify the path where you want to save the plot
    plt.show()


if __name__ == "__main__":
    main()