# ML4SCI DeepLense: Gravitational Lensing Substructure Detection

## Objective
The primary goal of this project is to leverage Deep Learning techniques, specifically Convolutional Neural Networks (CNNs), to automate the detection of substructured dark matter halos in strong gravitational lensing images. This task is crucial for understanding the nature of dark matter and its distribution within galaxies.

## Technical Approach
Our approach utilizes a modernized CNN architecture designed for astronomical image analysis:
- **Preprocessing**: Data is cast to `float32` to prevent overflow errors, followed by log-transformation to handle high dynamic range and Min-Max scaling for normalization.
- **Architecture**: A deep CNN incorporating `Conv2D` layers, `LeakyReLU` activations to prevent dead neurons in dark sky regions, `BatchNormalization` for training stability, and `GlobalAveragePooling2D` for efficient feature condensation.
- **Regularization**: Implementation of `Dropout` and `EarlyStopping` callbacks to prevent overfitting on the synthetic lensing dataset.
- **Experimental Transfer Learning**: An exploration into using a pre-trained `EfficientNetB0` backbone for improved feature extraction through transfer learning.

## Results Summary
The modernized CNN architecture demonstrates robust convergence and superior performance in distinguishing between "Normal" (smooth) and "Subhalo" (substructured) lensing profiles. Detailed training logs and validation metrics are provided within the notebook.

## Instructions
1. **Environment Setup**: Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execution**: Open and run `GSOC.ipynb` in a Jupyter environment or Google Colab.
3. **Data**: The notebook automatically downloads and extracts the required lensing dataset from the ML4SCI repository.
