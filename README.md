# Clustering-Based Coreset Construction for Efficient Deep Learning

This repository contains the implementation of a clustering-based coreset selection method developed as part of my Master's thesis. It includes scripts for constructing coresets and training a ResNet18 classification model on various datasets and data subsets (coresets, random subsets, or full datasets).

The current implementation supports the following datasets:
- ImageNet-VidVRD
- EchoNet-Dynamic
- EchoNet-Pediatric
- EchoNet-LVH

---

## Getting Started

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Jennyyy23/jennys_coreset_construction.git
cd jennys_coreset_construction
pip install -r requirements.txt
```

## Data Preprocessing
Scripts for preparing and preprocessing the datasets are located in the data_preprocessing/ folder.

## Coreset Construction

Coreset construction scripts are located in the `create_coresets/` folder.

- **Main script**: `jennys_coreset_construction.py`
- To generate coresets, label_csv files, and zipped archives in one go, use `all_methods_for_coreset_construction.py`.
- Alternatively, individual steps can be run using the respective scripts.

Additional scripts related to dataset distillation (based on [ultrasound_subset](https://github.com/Jack47744/ultrasound_subset)) can be found in the `dataset_distillation/` subfolder.

## Training
Training scripts, including WANDB hyperparameter sweep setup and the final training procedure for the ResNet18 classifier, are available in the training/ folder.