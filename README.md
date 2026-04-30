# Comparative Analysis of Machine Learning Algorithms for Breast Cancer Wisconsin dataset - Breast Cancer Prediction Project
This project focuses on building and evaluating machine learning models to predict breast cancer based on features extracted from digitized images of fine needle aspirate (FNA) of a breast mass. The goal is to classify a mass as either Benign (non-cancerous) or Malignant (cancerous).

Dataset
The project utilizes the Breast Cancer Wisconsin (Diagnostic) dataset, which contains 569 samples with 30 real-valued features computed from the digital images, along with an ID number and a diagnosis (M = Malignant, B = Benign).

Methodology

The following steps were performed in this analysis:

Data Loading & Initial Inspection: The dataset was loaded, and initial checks were performed to understand its structure, types, and presence of missing values.
Data Preprocessing:

Column Renaming: Features were assigned descriptive names based on the provided data dictionary.

Categorical Encoding: The 'Diagnosis' target variable ('M' and 'B') was converted into numerical format (1 for Malignant, 0 for Benign) and a human-readable label column was created for visualizations.

Feature Scaling: All numerical features were scaled using StandardScaler to ensure consistent ranges and prevent dominance by features with larger magnitudes.

Dimensionality Reduction (PCA):
Principal Component Analysis (PCA) was applied to reduce the feature space while retaining 95% of the variance, resulting in a more compact and potentially less noisy dataset for modeling.
A 2-component PCA was also performed for visualization purposes, allowing for a 2D representation of the data clusters.

Exploratory Data Analysis (EDA): Visualizations were created to understand the data distribution, correlations between features, and the relationship between key features and the target variable. This included a diagnosis distribution plot, a correlation heatmap, a boxplot comparing radius_mean across diagnoses, and a PCA visualization with eigenvectors and confidence ellipses.

Data Splitting: The preprocessed and dimensionality-reduced data was split into training (80%) and testing (20%) sets to ensure robust model evaluation.

Model Building & Training: Five popular classification algorithms were trained on the processed data:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Model Evaluation: Each model's performance was assessed using key metrics such as Accuracy, Precision, Recall, and F1-Score. Confusion matrices were generated to provide a detailed view of true positives, true negatives, false positives, and false negatives. Training times were also recorded for comparison.

Results

After evaluating all models, Logistic Regression emerged as the top-performing model, achieving the highest accuracy and F1-score on the test dataset. This suggests that the data, after PCA, is well-suited for a linear classification approach.

Further Work
To enhance this analysis and improve model robustness, future work could include:

--> K-fold Cross-Validation: Implement K-fold cross-validation for a more reliable estimate of model performance and reduced variance in evaluation metrics.

--> Deeper PCA Interpretation: Further analyze PCA loadings to gain deeper insights into which original features contribute most significantly to the principal components.


Technologies Used:
Python, 
Pandas (for data manipulation), 
NumPy (for numerical operations), 
Scikit-learn (for machine learning models, preprocessing, and evaluation), 
Matplotlib (for plotting), 
Seaborn (for advanced visualizations)
