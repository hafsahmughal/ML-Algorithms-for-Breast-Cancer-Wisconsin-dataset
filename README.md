# Comparative Analysis of Machine Learning Algorithms for Breast Cancer Wisconsin Dataset

## Overview
This project builds and evaluates machine learning models to predict breast cancer from features extracted from digitized fine needle aspirate (FNA) images. The goal is to classify tumors as either **Benign** or **Malignant**.

## Dataset
The analysis uses the Breast Cancer Wisconsin (Diagnostic) dataset, which contains:

- 569 samples
- 30 real-valued features computed from digitized images
- an ID number for each sample
- a diagnosis label: `M = Malignant`, `B = Benign`

## Methodology
The analysis follows these main steps:

### 1. Data Loading & Inspection
- Load the dataset
- Check structure, data types, and missing values

### 2. Data Preprocessing
- Rename columns using the provided data dictionary
- Encode the `Diagnosis` target as numeric values (`M = 1`, `B = 0`)
- Scale numerical features with `StandardScaler`

### 3. Dimensionality Reduction
- Apply Principal Component Analysis (PCA) to retain 95% of the variance
- Create a 2-component PCA projection for visualization

### 4. Exploratory Data Analysis (EDA)
Visualizations were created to explore feature relationships and diagnosis patterns:

- Diagnosis distribution plot
- Correlation heatmap
- Radius mean comparison by diagnosis
- PCA visualization with eigenvectors and confidence ellipses

Saved visualizations are available in the `images/` folder:

- `images/correlation_heatmap.png`
- `images/diagnosis_distribution.png`
- `images/pca_eigenvectors.png`
- `images/radius_mean_diagnosis.png`

### 5. Data Splitting
- Split data into training (80%) and testing (20%) sets

## Models
The following classification algorithms were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## Evaluation
Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Training time comparisons

## Results
Logistic Regression achieved the best overall performance, indicating that the PCA-reduced dataset is well-suited to a linear classifier.

## Future Work
Potential improvements include:

- Implementing K-fold cross-validation for more reliable model evaluation
- Performing a deeper analysis of PCA loadings to understand feature importance
- Exploring additional feature engineering and model tuning

## Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
