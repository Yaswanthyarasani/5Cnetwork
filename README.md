# Brain MRI Metastasis Segmentation

## Overview
This project compares Nested U-Net and Attention U-Net for brain metastasis segmentation on MRI images. A web application is provided to upload MRI images and visualize segmentation results.

## Steps to Run
1. Clone the repository:
2. Install the required libraries:
3. Run the training script (if not using pre-trained weights):
4. Start the FastAPI server:
5. Run the Streamlit application:
## Architectures
- **Nested U-Net (U-Net++)**: Adds dense skip connections for improved segmentation.
- **Attention U-Net**: Uses attention gates to focus on relevant regions in MRI images.

## Datasets
Images should be in the format: `TCGA_CS_<ID>_<sequence>.tif`. Ensure 80/20 split between train and test datasets.

## Results
Comparative DICE scores between the models:
- Nested U-Net: **XX.XX%**
- Attention U-Net: **XX.XX%**
