# Hyperparameter-Optimization-For-Xgboost

This notebook, Hyperparameter_Optimization_For_Xgboost.ipynb, provides a comprehensive guide on tuning a gradient boosting model to improve classification performance on a customer churn dataset.

What is happening in this code?
The notebook follows a standard machine learning pipeline focused on optimizing an XGBoost classifier:

Data Ingestion and Exploration:

It loads the Churn_Modelling.csv dataset, which contains customer information and a target variable indicating whether a customer stayed with or left a bank.

Feature Engineering:

It handles categorical data by converting the "Geography" and "Gender" columns into numerical dummy variables using one-hot encoding.

The features (X) and target labels (y) are separated, and the dataset is split into training and testing sets.

Hyperparameter Tuning:

The core of the notebook is the use of RandomizedSearchCV to find the best combination of XGBoost parameters.

It defines a "parameter grid" that explores various values for:

learning_rate: Step size shrinkage used in update to prevents overfitting.

max_depth: Maximum depth of a tree.

min_child_weight: Minimum sum of instance weight needed in a child.

gamma: Minimum loss reduction required to make a further partition on a leaf node.

colsample_bytree: Subsample ratio of columns when constructing each tree.

Cross-Validation:

It uses Stratified K-Fold Cross-Validation (via cross_val_score) to evaluate the model's robustness and ensure the accuracy scores are consistent across different subsets of the data.

What kind of results does it give?
Optimal Hyperparameters: The script identifies the specific parameters that yield the highest performance for the XGBoost model on this dataset.

Performance Metrics: It outputs the cross-validation scores for multiple folds (e.g., scores around 0.85 to 0.87) and calculates the mean accuracy (approximately 86.4%).
