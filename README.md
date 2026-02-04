# Student Placement and Salary Prediction using Machine Learning

## Project Overview

This project aims to analyze student academic and skill-related data to:

1. Predict whether a student gets placed (classification task)
2. Predict the expected salary for placed students (regression task)

The project demonstrates the complete machine learning pipeline, including data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

---

## Dataset

The dataset is sourced from Kaggle and contains information related to:

* Academic performance
* Skills and practical experience
* Demographic and background attributes
* Placement status and salary

Two CSV files were provided:

* One containing student attributes
* One containing placement status and salary details

These were merged using a unique student identifier.

**Dataset link:** [Indian Engineering College Placement Dataset](https://www.kaggle.com/datasets/vishardmehta/indian-engineering-college-placement-dataset)

---

## Problem Formulation

### 1. Placement Prediction (Classification)

* Target variable: `placement_status`
* Objective: Predict whether a student is placed or not

### 2. Salary Prediction (Regression)

* Target variable: `salary_lpa`
* Objective: Predict salary in LPA
* Important assumption: Salary prediction is performed **only for placed students**, as salary is undefined for non-placed students

This results in a two-stage modeling approach:

1. Placement classification
2. Salary regression conditional on placement

---

## Feature Engineering

Key steps included:

* Dropping weakly correlated and redundant features
* Combining related features into composite variables (e.g., skill ratings, practical experience)
* Encoding categorical variables
* Handling class imbalance using class weights
* Separating features used for classification and regression where appropriate

Final selected features include academic performance, CGPA, backlogs, skill ratings, practical experience, and key demographic indicators.

---

## Models Used

### Classification Model

* Random Forest Classifier
* Class imbalance handled using `class_weight='balanced'`
* Hyperparameter tuning using `RandomizedSearchCV`
* Evaluation metrics:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix

### Regression Model

* Random Forest Regressor
* Trained only on placed students
* Hyperparameter tuning using `RandomizedSearchCV`
* Evaluation metrics:

  * R² score
  * RMSE (Root Mean Squared Error)

---

## Results

### Placement Prediction

* Accuracy: ~0.88
* Improved recall for non-placed students after class balancing
* Strong overall F1-score despite class imbalance

### Salary Prediction

* R² score: ~0.75
* RMSE: ~3.05 LPA
* Indicates strong explanatory power with realistic prediction error

---

## Key Takeaways

* Accuracy alone is insufficient for imbalanced classification problems
* Proper problem framing significantly improves regression performance
* Separating placement and salary prediction leads to more realistic models
* Feature engineering and domain understanding have a large impact on results

---

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Jupyter Notebook

---

## How to Run

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Open the notebook and run cells sequentially

---

## Future Improvements

* Try gradient boosting models for salary prediction
* Explore interaction features
* Add company-level or college-tier data if available

---

## Author

Varun Shakya \
Machine Learning Student

