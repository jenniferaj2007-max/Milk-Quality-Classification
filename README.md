Milk Quality Classification System

Project Overview

The Milk Quality Classification System is a machine learning project that predicts milk quality based on its physicochemical properties.

The system classifies milk into three categories:

- Low
- Medium
- High

Four machine learning algorithms are implemented and evaluated to identify the best-performing classification model.

Dataset

- Source: "Milk Quality Dataset – Kaggle" (https://www.kaggle.com/datasets/cpluzshrijayan/milkquality)
- Original records: 1,059
- Features: 7
- Target column: Grade
- Classes: Low, Medium, High

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- VS Code

Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

Project Workflow

1. Data Loading
2. Data Understanding
3. Data Cleaning
4. Data Preprocessing
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Data Visualization
10. Feature Importance Analysis

Model Performance

Algorithm| Accuracy
Logistic Regression| 62.74%
Decision Tree| 97.17%
Random Forest| 88.21%
SVM| 95.28%

Best-performing model: Decision Tree, with 97.17% accuracy on the test set.

The dataset was split using stratified group splitting to keep identical feature records together. Scaling was fitted on the training data only.

Project Outputs

- Cleaned dataset
- Training and testing datasets
- Scaled feature datasets
- Saved machine learning models
- Model performance comparison CSV
- Model accuracy comparison graph
- Confusion matrix visualization
- Feature importance analysis

How to Run

1. Install the required libraries:

pip install pandas numpy scikit-learn matplotlib joblib

2. Place the downloaded "milknew.csv" dataset in the "Data/Archive" folder.

3. Run the Python files in this order:

python data_loading.py
python data_understanding.py
python data_cleaning.py
python data_preprocessing.py
python logistic_regression_model.py
python decision_tree_model.py
python random_forest_model.py
python svm_model.py
python model_comparison.py
python model_visualization.py
python confusion_matrix.py
python feature_importance.py

Conclusion

This project demonstrates how machine learning algorithms can be applied to milk quality classification. Four classification models were trained and compared. Among the evaluated models, Decision Tree achieved the highest observed test accuracy of 97.17%.

The project provides an end-to-end workflow covering data preparation, model development, evaluation, and visualization.