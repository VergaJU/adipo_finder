# adipo_finder

**Identify adipocytes in bone marrow spatial omics data from segmentation masks.**

Adipocytes are often lost in FFPE-processed bone marrow slides, leading to incomplete spatial proteomics or transcriptomics analysis. `adipo_finder` is a lightweight Python package designed to recover these regions by detecting adipocyte-shaped areas in segmentation masks, enabling more accurate tissue context reconstruction.

Now includes a **Machine Learning module** to filter false positives using shape, size, and spatial context features.

---

## 🚀 Features

- 🔬 Detects adipocytes based on shape and absence of marker staining
- 🧠 Compatible with segmentation masks from spatial omics pipelines
- 🧩 Integrates with `squidpy` and `AnnData` objects
- 🤖 **Machine Learning Classifier:** Train a PyTorch neural network to distinguish true adipocytes from artifacts.
- 📏 **Advanced Feature Extraction:** Uses shape descriptors (eccentricity, compactness), size, and distance from tissue boundaries.
- ⚙️ Customizable parameters for blurring, object size, and morphology
- 🧪 Includes a toy dataset for demonstration and testing

---

## 📦 Installation

From PyPI:

```bash
pip install adipo_finder
```

From GitHub:

```bash
pip install git+https://github.com/VergaJU/adipo_finder.git
```

**Requirements:** `numpy`, `pandas`, `scanpy`, `squidpy`, `matplotlib`, `anndata`, `scikit-image`, `scipy`, `scikit-learn`, `torch`, `Pillow`.

---

## 🧰 Usage

### Minimal example (Segmentation)

```python
from adipo_finder import utils as seg_utils
from adipo_finder import segmentation as seg
import matplotlib.pyplot as plt
import skimage.io as io


segmentation = io.imread("path_to_mask.tif")

old_seg,segmentation=seg_utils.Preprocessing.preprocess_image(image=segmentation)
new_img=seg.Segmentation.run_segmentation(image=img)
new_seg,df=seg_utils.Exporting.export_adipocytes(segmentation_image=old_seg,new_segmentation=new_img)
plt.imshow(new_seg)
```

### With AnnData (Squidpy integration)

```python
from adipo_finder import utils as seg_utils
from adipo_finder import segmentation as seg
import matplotlib.pyplot as plt
import anndata as ad

adata = ad.read("path_to_anndata.h5ad")

old_seg,segmentation=seg_utils.Preprocessing.preprocess_image(adata=adata, library_id="library_identifier")
new_img=seg.Segmentation.run_segmentation(image=img)
new_seg,df=seg_utils.Exporting.export_adipocytes(segmentation_image=old_seg,new_segmentation=new_img)

df_exp= seg_utils.Exporting.expand_df(adata=adata,df=df, library_id="library_identifier")
adata_adipo = seg_utils.Exporting.create_adipo_adata(adata=adata,df=df_exp) 
adata_new=seg_utils.Exporting.merge_adatas(adata=adata,adata_tmp=adata_adipo,new_segmentation=new_seg,library_id="library_identifier")
```

### Machine Learning Classification

You can now train a classifier to improve precision by filtering out false positives.

```python
from adipo_finder import AdipoModel, FeatureExtraction, Segmentation
import pandas as pd

# 1. Preprocess and Extract Features
# Assume you have a list of images and optionally ground truth masks
# ... (load images) ...

# 2. Prepare Data (Example)
# df = ... (dataframe with features extracted via FeatureExtraction.calculate_features)

# 3. Train Model
model, train_ids, val_ids, test_ids, scaler = AdipoModel.train_model(df, n_epochs=500)

# 4. Predict and Clean Image
cleaned_seg = AdipoModel.predict_and_clean_image(model, scaler, image_id, df, original_segmentation)
```

## 📂 Toy Dataset

A toy image is included in the repository to demonstrate how `adipo_finder` works. Use this for quick testing or to tune parameters before applying to real data.


## ⚙️ Parameters


| Parameter        | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `sigma`          | Gaussian blur standard deviation (default: 2)             |
| `min_size`       | Minimum size (in pixels) for adipocyte regions            |
| `max_size`       | Maximum size (in pixels) for adipocyte regions            |
| `opening_window` | Size of structuring element used in morphological opening |
| `expand_pixels`  | Number of pixels to expand adipocyte mask after detection |


## 📖 Documentation

Full documentation and examples are available in the GitHub Repository


## 🧪 Citation

If you use adipo_finder in your work, please cite the corresponding AppNote or preprint (coming soon).


## 🛠️ License

This project is licensed under the MIT License.


✨ Contributions

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.
