# Heart-disease-prediction-using-Machine-Learning
Heart Disease Prediction using Machine Learning (Logistic Regression)
Project Overview
This project aims to predict the likelihood of heart disease in a patient using key medical indicators.
Logistic Regression, a classification algorithm, is used due to its effectiveness in binary outcome prediction.
Dataset
Source: Kaggle Heart Disease Dataset
Features: 13 medical attributes such as age, sex, cholesterol level, resting blood pressure, and more.
Target: Presence (1) or Absence (0) of heart disease.
Prerequisites
Python 3.x
Libraries:
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook (optional)
Project Structure
data/: Contains the heart disease dataset (CSV file).
notebooks/: Jupyter notebooks with model code and analysis.
models/: Saved Logistic Regression models.
scripts/: Python scripts for data processing and model building.
README.md: Documentation of the project.
Steps Involved
Data Preprocessing

Load and inspect the dataset.
Handle missing values (if any).
Perform data normalization or standardization.
Feature selection (if necessary) to improve model performance.
Exploratory Data Analysis (EDA)

Visualize key features using histograms, boxplots, and correlation heatmaps.
Analyze class distributions for heart disease presence/absence.
Look for feature relationships and patterns relevant to the target variable.
Splitting Data

Split the data into training and testing sets using train_test_split.
Common split: 80% training and 20% testing.
Model Building

Import Logistic Regression from sklearn.linear_model.
Fit the model using the training data.
Set up hyperparameters (e.g., C, solver, max_iter).
Model Evaluation

Evaluate the model using:
Accuracy
Confusion matrix
Precision, Recall, F1 Score
ROC-AUC curve
Perform k-fold cross-validation to check model stability.
Model Optimization

Tune hyperparameters to improve performance using GridSearchCV.
Consider regularization techniques (L2) to reduce overfitting.
Predictions

Predict heart disease on the test data.
Use the model to predict the likelihood of heart disease in new patients.
Saving the Model

Save the trained Logistic Regression model using joblib or pickle for future predictions.
Results
The Logistic Regression model provides a balanced accuracy in predicting heart disease.
The project achieved a ROC-AUC score of XX and accuracy of YY%.
