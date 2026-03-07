import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error, f1_score, accuracy_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_and_preprocess_data():
    """Load and preprocess the student placement data"""
    # Load data
    student = pd.read_csv('data/student.csv')
    placement = pd.read_csv('data/placement.csv')

    # Merge datasets
    df = pd.merge(student, placement, on='Student_ID', how='inner')

    # Handle missing values
    df['extracurricular_involvement'] = df['extracurricular_involvement'].fillna(df['extracurricular_involvement'].mode()[0])

    # Encode categorical variables
    job_map = {'Placed': True, 'Not Placed': False}
    df['placement_status'] = df['placement_status'].map(job_map).astype(int)

    ord_map = {
        'gender': {'Male': 1, 'Female': 0},
        'part_time_job': {'Yes': 1, 'No': 0},
        'internet_access': {'Yes': 1, 'No': 0},
        'family_income_level': {'Medium': 2, 'Low': 1, 'High': 3},
        'city_tier': {'Tier 2': 2, 'Tier 3': 3, 'Tier 1': 1},
        'extracurricular_involvement': {'Medium': 2, 'Low': 1, 'High': 3}
    }

    for col, mapp in ord_map.items():
        df[col] = df[col].map(mapp).astype('int')

    # One-hot encode branch
    df_encoded = pd.get_dummies(df, columns=['branch'])

    # Feature engineering
    df_encoded['practical_experience'] = (df_encoded['projects_completed'] +
                                        df_encoded['internships_completed'] +
                                        df_encoded['hackathons_participated'])
    df_encoded["skill_rating"] = (0.5*df_encoded["coding_skill_rating"] +
                                0.2*df_encoded["communication_skill_rating"] +
                                0.3*df_encoded["aptitude_skill_rating"])

    # Drop unnecessary columns
    cols_to_drop = ['tenth_percentage', 'study_hours_per_day', 'attendance_percentage',
                   'projects_completed', 'internships_completed', 'coding_skill_rating',
                   'communication_skill_rating', 'aptitude_skill_rating',
                   'hackathons_participated', 'sleep_hours', 'stress_level',
                   'Student_ID', 'certifications_count']

    df_encoded = df_encoded.drop(columns=cols_to_drop)

    return df_encoded

def train_models(df):
    """Train both classification and regression models"""

    # Classification model
    x = df.drop(['placement_status', 'salary_lpa'], axis=1)
    y = df['placement_status']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=67, stratify=y)

    rf_clf = RandomForestClassifier(random_state=67, class_weight='balanced')

    params_clf = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rand_clf = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=params_clf,
        n_iter=40,
        cv=5,
        scoring='f1_macro',
        random_state=67,
        n_jobs=-1
    )

    rand_clf.fit(x_train, y_train)
    best_clf = rand_clf.best_estimator_

    # Regression model (only for placed students)
    df_placed = df[df['placement_status'] == 1]
    u = df_placed.drop(['placement_status', 'salary_lpa'], axis=1)
    v = df_placed['salary_lpa']

    u_train, u_test, v_train, v_test = train_test_split(u, v, test_size=0.2, random_state=67)

    rf_reg = RandomForestRegressor(random_state=67)

    params_reg = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rand_reg = RandomizedSearchCV(
        estimator=rf_reg,
        param_distributions=params_reg,
        n_iter=40,
        cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=67,
        n_jobs=-1
    )

    rand_reg.fit(u_train, v_train)
    best_reg = rand_reg.best_estimator_

    return best_clf, best_reg, x_train.columns.tolist()

if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data()

    # Train models
    clf_model, reg_model, feature_names = train_models(df)

    # Save models
    joblib.dump(clf_model, 'models/placement_classifier.pkl')
    joblib.dump(reg_model, 'models/salary_regressor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print("Models trained and saved successfully!")