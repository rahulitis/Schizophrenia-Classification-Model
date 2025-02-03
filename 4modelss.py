import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss,precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression

RandomForest = RandomForestClassifier(random_state=1, n_estimators=100)
LogisticRegression = LogisticRegression(random_state=3, max_iter=2000)

# Load the data
data = pd.read_csv(r"C:\Users\HomePC\Desktop\psykose\schizophrenia-features.csv")

# Select relevant features
features = ['f.mean', 'f.sd', 'f.propZeros']
X = data[features]
y = data['class'] 

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4)


# Define numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier',RandomForest)])

# Fit the model
model = Pipeline(steps=[('preprocessor', preprocessor),  # Replace 'preprocessor' with your actual preprocessing step
                        ('classifier', RandomForestClassifier(random_state=1))])

# Fit the model
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_valid)
y_pred_proba = model.predict_proba(X_valid)[:, 1]
print("Accuracy:", accuracy_score(y_valid, preds))
print("Log Loss:", log_loss(y_valid, y_pred_proba))

# Cross-validation
folding = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
scores = cross_validate(model, X, y, cv=folding, scoring=scoring_metrics, return_train_score=False)

print("Cross-validated scores:")
for metric in scoring_metrics:
    print(f"{metric}: {scores['test_' + metric].mean()} (std: {scores['test_' + metric].std()})")

# Variables to store cumulative results for plots
fprs = []
tprs = []
precisions = []
recalls = []
pr_auc_scores = []
roc_auc_scores = []

# Calculate Precision-Recall and ROC curves for each fold in cross-validation
for train_index, test_index in folding.split(X, y):
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train_fold, y_train_fold)
    y_pred_proba = model.predict_proba(X_valid_fold)[:, 1]


    precision, recall, _ = precision_recall_curve(y_valid_fold, y_pred_proba)
    precisions.append(precision)
    recalls.append(recall)
    pr_auc_scores.append(auc(recall, precision))


    fpr, tpr, _ = roc_curve(y_valid_fold, y_pred_proba)
    fprs.append(fpr)
    tprs.append(tpr)
    roc_auc_scores.append(auc(fpr, tpr))

avg_pr_auc = np.mean(pr_auc_scores)
avg_roc_auc = np.mean(roc_auc_scores)

# Plot Precision-Recall Curve and ROC Curve
plt.figure(figsize=(12, 5))

# Plot Precision-Recall Curve
plt.subplot(1, 2, 1)
for precision, recall in zip(precisions, recalls):
    plt.plot(recall, precision, color='blue', lw=2, alpha=0.3)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Avg AUC = {avg_pr_auc:.2f})')

# Plot ROC Curve
plt.subplot(1, 2, 2)
for fpr, tpr in zip(fprs, tprs):
    plt.plot(fpr, tpr, color='red', lw=2, alpha=0.3)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (Avg AUC = {avg_roc_auc:.2f})')

plt.tight_layout()
plt.show()



# Define KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=1)

# For XGBoost
XGbb = XGBClassifier(random_state=2, n_estimators=100)

# Initialize lists for storing scores for XGBoost
accuracies_xgb = []
log_losses_xgb = []
precisions_xgb = []
recalls_xgb = []
roc_aucs_xgb = []

# XGBoost Cross-validation
for train_index, test_index in k_fold.split(X):
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[test_index]
    XGbb.fit(X_train_fold, y_train_fold)
    y_pred = XGbb.predict(X_valid_fold)
    y_pred_proba = XGbb.predict_proba(X_valid_fold)[:, 1]

    accuracies_xgb.append(accuracy_score(y_valid_fold, y_pred))
    log_losses_xgb.append(log_loss(y_valid_fold, y_pred_proba))
    
    precision, recall, _ = precision_recall_curve(y_valid_fold, y_pred_proba)
    precisions_xgb.append(precision)
    recalls_xgb.append(recall)
    
    fpr, tpr, _ = roc_curve(y_valid_fold, y_pred_proba)
    roc_aucs_xgb.append(auc(fpr, tpr))

# Print results for XGBoost
print("XGBoost Accuracy:", np.mean(accuracies_xgb))
print("XGBoost Log Loss:", np.mean(log_losses_xgb))
print("XGBoost ROC AUC:", np.mean(roc_aucs_xgb))

# Plot Precision-Recall Curve for XGBoost
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for precision, recall in zip(precisions_xgb, recalls_xgb):
    plt.plot(recall, precision, color='blue', lw=2, alpha=0.3)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (XGBoost)')

# Plot ROC Curve for XGBoost
plt.subplot(1, 2, 2)
for fpr, tpr in zip(precisions_xgb, recalls_xgb):
    plt.plot(fpr, tpr, color='red', lw=2, alpha=0.3)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (XGBoost)')

plt.tight_layout()
plt.show()


LGBM = lgb.LGBMClassifier(random_state=3, n_estimators=100) 

# Initialize lists for storing scores for LightGBM
accuracies_lgbm = []
log_losses_lgbm = []
precisions_lgbm = []
recalls_lgbm = []
roc_aucs_lgbm = []

# LightGBM Cross-validation
for train_index, test_index in k_fold.split(X):
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[test_index]
    LGBM.fit(X_train_fold, y_train_fold)
    y_pred = LGBM.predict(X_valid_fold)
    y_pred_proba = LGBM.predict_proba(X_valid_fold)[:, 1]

    accuracies_lgbm.append(accuracy_score(y_valid_fold, y_pred))
    log_losses_lgbm.append(log_loss(y_valid_fold, y_pred_proba))
    
    precision, recall, _ = precision_recall_curve(y_valid_fold, y_pred_proba)
    precisions_lgbm.append(precision)
    recalls_lgbm.append(recall)
    
    fpr, tpr, _ = roc_curve(y_valid_fold, y_pred_proba)
    roc_aucs_lgbm.append(auc(fpr, tpr))

# Print results for LightGBM
print("LightGBM Accuracy:", np.mean(accuracies_lgbm))
print("LightGBM Log Loss:", np.mean(log_losses_lgbm))
print("LightGBM ROC AUC:", np.mean(roc_aucs_lgbm))

# Plot Precision-Recall Curve for LightGBM
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for precision, recall in zip(precisions_lgbm, recalls_lgbm):
    plt.plot(recall, precision, color='blue', lw=2, alpha=0.3)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (LightGBM)')

# Plot ROC Curve for LightGBM
plt.subplot(1, 2, 2)
for fpr, tpr in zip(precisions_lgbm, recalls_lgbm):
    plt.plot(fpr, tpr, color='red', lw=2, alpha=0.3)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (LightGBM)')

plt.tight_layout()
plt.show()




# Logistic Regression
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state=3, max_iter=2000)

accuracies_LogReg = []
log_losses_LogReg = []
precisions_LogReg = []
recalls_LogReg = []
roc_aucs_LogReg = []

fprs = []  # To store all false positive rates (fpr)
tprs = []  # To store all true positive rates (tpr)

# Logistic Regression Cross-validation
for train_index, test_index in k_fold.split(X):
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[test_index]
    LogReg.fit(X_train_fold, y_train_fold)
    y_pred = LogReg.predict(X_valid_fold)
    y_pred_proba = LogReg.predict_proba(X_valid_fold)[:, 1]

    accuracies_LogReg.append(accuracy_score(y_valid_fold, y_pred))
    log_losses_LogReg.append(log_loss(y_valid_fold, y_pred_proba))
    
    precision, recall, _ = precision_recall_curve(y_valid_fold, y_pred_proba)
    precisions_LogReg.append(precision)
    recalls_LogReg.append(recall)
    
    fpr, tpr, _ = roc_curve(y_valid_fold, y_pred_proba)
    fprs.append(fpr)
    tprs.append(tpr)
    roc_aucs_LogReg.append(auc(fpr, tpr))

# Print results for Logistic Regression
print("Logistic Regression Accuracy:", np.mean(accuracies_LogReg))
print("Logistic Regression Log Loss:", np.mean(log_losses_LogReg))
print("Logistic Regression ROC AUC:", np.mean(roc_aucs_LogReg))

# Plot Precision-Recall Curve for Logistic Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for precision, recall in zip(precisions_LogReg, recalls_LogReg):
    plt.plot(recall, precision, color='blue', lw=2, alpha=0.3)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Logistic Regression)')

# Plot ROC Curve for Logistic Regression
plt.subplot(1, 2, 2)
for fpr, tpr in zip(fprs, tprs):  # Plot all FPR and TPR for each fold
    plt.plot(fpr, tpr, color='red', lw=2, alpha=0.3)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Logistic Regression)')

plt.tight_layout()
plt.show()
