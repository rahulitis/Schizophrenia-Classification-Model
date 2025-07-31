# Schizophrenia Classification Model

## Overview
This project implements a machine learning pipeline to classify schizophrenia using features from a dataset. It employs multiple classifiers (Random Forest, XGBoost, LightGBM, and Logistic Regression) to predict the class labels and evaluates their performance using various metrics such as accuracy, log loss, precision, recall, and ROC AUC. The code includes data preprocessing, model training, cross-validation, and visualization of Precision-Recall and ROC curves.
Dataset
The dataset used is schizophrenia-features.csv, which contains features related to schizophrenia classification. The relevant features used for modeling are:

f.mean: Mean of the feature values
f.sd: Standard deviation of the feature values
f.propZeros: Proportion of zero values in the feature

The target variable is class, which represents the classification label (e.g., positive or negative for schizophrenia).

# Requirements
To run the code, the following Python libraries are required:

**pandas**
**numpy**
**matplotlib**
**xgboost**
**scikit-learn**
**lightgbm**

Install the dependencies using:
```
pip install pandas numpy matplotlib xgboost scikit-learn lightgbm
```
## Code Structure
The code is organized into the following sections:

**Data Loading and Preprocessing**:

+ Loads the dataset using pandas.
+ Selects relevant features (f.mean, f.sd, f.propZeros) and the +target variable (class).
+ Splits the data into training (80%) and validation (20%) sets.
+ Applies preprocessing using sklearn's ColumnTransformer to handle +numerical and categorical features:
+ Numerical features: Imputation with mean and standard scaling.
+ Categorical features: Imputation with most frequent value and one-+hot encoding.




**Model Definitions**:

Random Forest: Configured with 100 estimators and a random state of 
+ XGBoost: Configured with 100 estimators and a random state of
+ LightGBM: Configured with 100 estimators and a random state of 
+ Logistic Regression: Configured with a maximum of 2000 iterations and a random state of 3.


**Model Training and Evaluation**:

A pipeline is created combining preprocessing and the classifier.
The model is trained on the training set and evaluated on the validation set using accuracy and log loss.
Cross-validation (10-fold stratified) is performed to evaluate metrics: accuracy, precision, recall, and F1-score.
Precision-Recall and ROC curves are generated for each fold, with average AUC scores calculated.


**Cross-Validation for Other Models**:

Separate cross-validation loops are implemented for XGBoost, LightGBM, and Logistic Regression.
Metrics (accuracy, log loss, ROC AUC) are computed and stored for each fold.
Precision-Recall and ROC curves are plotted for each model.


**Visualization**:

Precision-Recall and ROC curves are visualized for each model across all folds.
Plots include a diagonal line for ROC curves to represent a random classifier.



## Usage

Ensure the dataset schizophrenia-features.csv is available in the specified directory or update the file path in the code.
Run the script in a Python environment with the required libraries installed.
The script will:
Train and evaluate the Random Forest model on the train/validation split.
Perform 10-fold cross-validation for Random Forest, XGBoost, LightGBM, and Logistic Regression.
Output performance metrics (accuracy, log loss, ROC AUC) for each model.
Generate and display Precision-Recall and ROC curves for each model.



**Output**

The script produces the following outputs:

Console Output:
Accuracy and log loss for the Random Forest model on the validation set.
Cross-validated metrics (accuracy, precision, recall, F1-score) for Random Forest.
Mean accuracy, log loss, and ROC AUC for XGBoost, LightGBM, and Logistic Regression.


Visualizations:
Precision-Recall curves for each model, showing performance across folds.
ROC curves for each model, with a diagonal line for reference and average AUC scores.



Notes

The dataset path (C:\Users\HomePC\Desktop\psykose\schizophrenia-features.csv) should be updated to match the local environment.
The random state values ensure reproducibility of results.
The code assumes numerical features in the dataset; categorical feature preprocessing is included but may not be used if no categorical features are present.
The ROC curve plotting for XGBoost and LightGBM contains an error in the original code: it incorrectly uses precisions_xgb and recalls_xgb (or precisions_lgbm and recalls_lgbm) instead of fprs and tprs. This should be corrected for accurate ROC curve visualization.

```
Results
Accuracy: 0.8333333333333334
Log Loss: 0.3718840316925055
Cross-validated scores:
accuracy: 0.8341219096334186 (std: 0.029522452862097856)
precision_weighted: 0.8361378048571071 (std: 0.02955089172570698)
recall_weighted: 0.8341219096334186 (std: 0.029522452862097856)
f1_weighted: 0.8326883044514851 (std: 0.03038175562014505)
XGBoost Accuracy: 0.8297953964194372
XGBoost Log Loss: 0.4508620989517837
XGBoost ROC AUC: 0.9161682593148497
LightGBM Accuracy: 0.8326513213981244
LightGBM Log Loss: 0.4564622777584413
LightGBM ROC AUC: 0.9168464821826806
Logistic Regression Accuracy: 0.8485720375106564
Logistic Regression Log Loss: 0.3808506579324614
Logistic Regression ROC AUC: 0.9122030806710735
```
## Limitations

The code assumes the dataset is clean and properly formatted.
The preprocessing pipeline may need adjustments if additional feature types (e.g., categorical features) are introduced.
The ROC curve plotting issue mentioned above should be addressed for accurate visualization.

## Future Improvements

Correct the ROC curve plotting for XGBoost and LightGBM by using fprs and tprs.
Add hyperparameter tuning (e.g., using GridSearchCV) to optimize model performance.
Include feature importance analysis to understand which features contribute most to predictions.
Save trained models and evaluation metrics for reproducibility and deployment.


