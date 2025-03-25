# Tumor Classification Project

This project implements a neural network-based classifier for tumor classification using PyTorch. The model is designed to classify breast tumors as either malignant (cancerous) or benign (non-cancerous) based on various patient and tumor characteristics.

## Description

This project focuses on developing a machine learning model to assist in the early detection and classification of breast tumors. The model uses a neural network architecture to analyze various features of breast cancer cases, including patient demographics, tumor characteristics, and medical indicators.

### Key Features
- Binary classification of breast tumors (malignant vs. benign)
- Uses 30 different features for prediction
- Implements a neural network with multiple layers
- Includes data preprocessing and standardization
- Provides training progress monitoring and accuracy metrics
- Interactive prediction interface for new samples

### Technical Details
The model processes numerical data from medical records and learns patterns that help distinguish between malignant and benign tumors. It uses a combination of:
- StandardScaler for feature normalization
- PyTorch's neural network framework
- Binary Cross-Entropy loss function
- Adam optimizer for training
- ReLU activation functions for hidden layers
- Sigmoid activation for final classification

### Use Case
This model can be used as a preliminary screening tool to assist medical professionals in the early stages of breast cancer diagnosis. However, it should be noted that this is a simplified model and should not be used as the sole basis for medical diagnosis.

## Project Structure

```
.
├── src/              # Source code
│   ├── models/      # Model definitions
│   │   └── tumor_classifier.py
│   ├── train.py     # Training script
│   └── predict.py   # Prediction script
├── models/          # Saved models directory
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - For Windows PowerShell:
   ```bash
   .\venv\Scripts\activate
   ```
   - For Windows Command Prompt (cmd):
   ```bash
   venv\Scripts\activate.bat
   ```
   - For Unix/MacOS:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model:
```bash
python src/train.py
```

The script will:
- Load and preprocess the data from scikit-learn's breast cancer dataset
- Train the neural network
- Print training progress every 10 epochs
- Display the final test accuracy
- Show detailed classification metrics including:
  - Overall accuracy
  - Precision, recall, and F1-score for each class
  - Confusion matrix
  - Distribution of malignant and benign tumors
- Save the trained model and scaler to the 'models' directory

### Making Predictions
To make predictions on new samples:
```bash
python src/predict.py
```

The prediction script will:
- Load the trained model and scaler
- Prompt you to enter the 30 features for a new sample
- Display the prediction (Benign or Malignant) and confidence score
- Allow you to make multiple predictions

## Model Architecture

The model uses a simple feed-forward neural network with:
- Input layer: 30 features from the breast cancer dataset
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

## Dataset

The model uses scikit-learn's built-in breast cancer dataset, which includes:
- 30 numerical features describing various characteristics of breast cancer cases
- Binary target variable (0 for malignant, 1 for benign)
- 569 samples in total
- Features include mean, standard error, and worst values of various tumor measurements
