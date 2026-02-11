import scanpy as sc
import numpy as np 
from anndata import AnnData
import anndata as ad
from matplotlib import pyplot as plt 
from skimage import measure, filters
import pandas as pd
from PIL import Image

class Preprocessing:
    @staticmethod
    def extract_segmentation_image(adata: AnnData,
                                library_id: str,
                                spatial_key: str = 'spatial',
                                image_key: str = 'images',
                                segmentation_key: str = 'segmentation', **kwargs) -> np.ndarray:
        """
        Extract segmentation image from AnnData object as np.ndarray.
        Specify the library_id, image_key and segmentation_key.

        parameters:
        adata: AnnData object
        spatial_key: str, key of the spatial coordinates in adata.uns
        library_id: str, library id of the image in adata.uns[spatial_key]
        image_key: str, key of the image in adata.uns[spatial_key][library_id]
        segmentation_key: str, key of the segmentation adata.uns[spatial_key][library_id][image_key]

        return:
        np.ndarray, segmentation image
        """
        img=adata.uns[spatial_key][library_id][image_key][segmentation_key].copy()
        return img

    @staticmethod
    def segmentation_to_binary(segmentation: np.ndarray, threshold: float = 0.5,**kwargs) -> np.ndarray:
        """
        Convert segmentation image to binary mask.

        parameters:
        segmentation: np.ndarray, segmentation image
        threshold: float, threshold to binarize the segmentation image

        return:
        np.ndarray, binary mask
        """
        return (255 * (segmentation > threshold))


    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0,**kwargs) -> np.ndarray:
        """
        Apply Gaussian filter to the image.

        parameters:
        image: np.ndarray, input image
        sigma: float, standard deviation of the Gaussian filter

        return:
        np.ndarray, filtered image
        """
        if sigma is not None:
            image = filters.gaussian(image, sigma=sigma,preserve_range=True).astype('uint8')
        return image


    @staticmethod
    def invert_image(image: np.ndarray) -> np.ndarray:
        """
        Invert the image.

        parameters:
        image: np.ndarray, input image

        return:
        np.ndarray, inverted image
        """
        return np.where(image == 0, 255, 0).astype('uint8')


    @classmethod
    def preprocess_image(cls, image: np.ndarray = None, adata: AnnData = None,library_id: str = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image for segmentation.
        Extract the segmentation image from the AnnData object.
        Convert the segmentation image to binary mask.
        Apply Gaussian filter to the binary mask.
        Invert the binary mask.

        parameters:
        adata: AnnData object
        library_id: str, library id of the image in adata.uns[spatial_key]

        return:
        tuple[np.ndarray, np.ndarray], (segmentation_image, inverted_image)
        """
        # set kwargs for the functions
        spatial_key = kwargs.get('spatial_key', 'spatial')
        image_key = kwargs.get('image_key', 'images')
        segmentation_key = kwargs.get('segmentation_key', 'segmentation')
        threshold = kwargs.get('threshold', 0.5)
        sigma = kwargs.get('sigma', 1.0)
        # Extract the segmentation image
        if adata is not None:
            segmentation_image = cls.extract_segmentation_image(adata, library_id, **kwargs)
        elif image is not None:
            segmentation_image = image
        else:
            raise ValueError("Either adata or image must be provided.")
        # apply gaussian filter
        blurred_image = cls.apply_gaussian_filter(segmentation_image,**kwargs)
        # convert to binary
        binary_image = cls.segmentation_to_binary(blurred_image,**kwargs)
        # invert image
        inverted_image = cls.invert_image(binary_image)
        return segmentation_image,inverted_image


class Exporting:
    @staticmethod
    def export_adipocytes(segmentation_image: np.ndarray, 
                        new_segmentation: np.ndarray, 
                        cell_id: str = 'Cell_ID', **kwargs) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Export adipocytes from the segmentation image.
        Add the new objects to the original segmentation image, coloring maintain the same order.
        Export objects numbers and metadata as a pandas dataframe to add the to the single cell object.

        parameters:
        segmentation_image: np.ndarray, segmentation image
        new_segmentation: np.ndarray, new segmentation image

        return:
        np.ndarray, updated segmentation image
        """
        base_image=segmentation_image.copy()
        # Initialize the new object value
        new_value = segmentation_image.max()+1
        # DataFrame to store object properties
        columns = ['area', 'centroid-0', 'centroid-1', 'axis_major_length', 'axis_minor_length', 'eccentricity', cell_id]
        df = pd.DataFrame(columns=columns)

        # Update the base image and extract object properties
        for region in measure.regionprops(new_segmentation):
            base_image[new_segmentation == region.label] = new_value
            new_row=pd.DataFrame.from_dict({
                'area': region.area,
                'centroid-0': region.centroid[0],
                'centroid-1': region.centroid[1],
                'axis_major_length': region.major_axis_length,
                'axis_minor_length': region.minor_axis_length,
                'eccentricity': region.eccentricity,
                cell_id: new_value
            },
                                        orient='index').T
            df = pd.concat([df,new_row], axis=0, ignore_index=True)
            new_value += 1
        
        return base_image, df


    @staticmethod
    def expand_df(adata: AnnData, df: pd.DataFrame, 
                library_id: str, 
                missing_columns: list[str],
                cell_annot_key: str | list[str],
                cell_type: str = 'Adipocyte',
                cell_id: str = 'Cell_ID',
                library_key: str = 'library_id', **kwarg) -> pd.DataFrame:
        """
        Integrate the adipocytes metadata with the AnnData obs.

        parameters:
        adata: AnnData object
        df: pd.DataFrame, adipocytes data
        library_id: str, library id of the image in adata.uns[spatial_key]
        missing_columns: list, missing columns to add to the adata.obs
        cell_id: str, key of the cell id in adata.obs
        library_key: str, key of the library id in adata.obs


        return:
        AnnData, updated AnnData object
        """
        # retrieve info from adata.obs
        missing_data=adata.obs[adata.obs[library_key]==library_id][missing_columns].iloc[0].to_dict()
        df.index='s'+library_id.split('_')[1]+'_'+df[cell_id].astype(int).astype(str)
        df[cell_id]=df[cell_id].astype(np.int64)
        df['ObjectNumber']=df[cell_id].astype(np.int64)
        df[library_key]=library_id
        if isinstance(cell_annot_key, str):
            df[cell_annot_key]=cell_type
        else:
            for i in cell_annot_key:
                df[i]=cell_type
        for k,v in missing_data.items():
            df[k]=v
        return df

    # function that takes the adata and the df and create new adata with the new df
    @staticmethod
    def create_adipo_adata(adata: AnnData,
                            df: pd.DataFrame, 
                            spatial_key: str = 'spatial',**kwargs) -> AnnData:
        """
        Create a new AnnData object with the adipocytes metadata.

        parameters:
        adata: AnnData object
        df: pd.DataFrame, adipocytes data
        spatial_key: str, key of the spatial coordinates in adata.obsm

        return:
        AnnData, new AnnData object
        """
        # create numpy array with centroids for spatial analysies
        spatial_tmp=df[['centroid-0','centroid-1']].to_numpy()
        # create numpy array of zeros for expression values
        # TODO: mean expression values from the channels?
        counts_tmp=np.zeros((df.shape[0],adata.shape[1]))
        # create adata object, fill it with numpy arra, spatial features, metadata and var_names
        adata_tmp=ad.AnnData(counts_tmp)
        adata_tmp.obs=df
        adata_tmp.obsm[spatial_key]=spatial_tmp
        adata_tmp.obs_names=list(df.index)
        adata_tmp.var_names=adata.var_names
        return adata_tmp
    

    # Function that takes the two adata and the new segmentation image and update the adata with the new segmentation
    def merge_adatas(adata: AnnData, 
                    adata_tmp: AnnData, 
                    new_segmentation: np.ndarray, 
                    library_id: str, 
                    spatial_key: str = 'spatial',
                    image_key: str = 'images',
                    segmentation_key: str = 'segmentation', **kwargs) -> AnnData:
        """
        Merge the new AnnData object with the original AnnData object.

        parameters:
        adata: AnnData object
        adata_tmp: AnnData object
        new_segmentation: np.ndarray, new segmentation image
        library_id: str, library id of the image in adata.uns[spatial_key]

        return:
        AnnData, updated AnnData object
        """
        # merge adata, outer join and keep uns
        adata_new=ad.concat([adata,adata_tmp],join='outer',uns_merge='only')
        # update the segmentation image
        adata_new.uns[spatial_key][library_id][image_key][segmentation_key]=new_segmentation
        return adata_new

def shuffle_labels(labeled_image: np.ndarray) -> np.ndarray:
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

def remove_segments(segmented: np.ndarray, remove_ids: list[int]) -> np.ndarray:
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

def process_ground_truth_image(base_path: str, id: str) -> None:
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
