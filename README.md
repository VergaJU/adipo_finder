# adipo_finder

**Identify adipocytes in bone marrow spatial omics data from segmentation masks.**

Adipocytes are often lost in FFPE-processed bone marrow slides, leading to incomplete spatial proteomics or transcriptomics analysis. `adipo_finder` is a lightweight Python package designed to recover these regions by detecting adipocyte-shaped areas in segmentation masks, enabling more accurate tissue context reconstruction.

---

## 🚀 Features

- 🔬 Detects adipocytes based on shape and absence of marker staining
- 🧠 Compatible with segmentation masks from spatial omics pipelines
- 🧩 Integrates with `squidpy` and `AnnData` objects
- ⚙️ Customizable parameters for blurring, object size, and morphology
- 🧪 Includes a toy dataset for demonstration and testing

---

## 📦 Installation

From PyPI:

```bash
pip install adipo_finder
