# Student Placement & Salary Prediction

A machine learning project that analyzes student academic and skill-based data to predict **placement outcomes** and **expected salary packages**.

This project demonstrates a **complete machine learning pipeline**, including data preprocessing, model training, evaluation, and deployment using an interactive web application.

---

# Project Overview

This system predicts two key outcomes for students:

**1. Placement Prediction (Classification)**
Determines whether a student is likely to get placed based on academic performance, skills, and background.

**2. Salary Prediction (Regression)**
Estimates the expected salary (in LPA) for students predicted to be placed.

An interactive **Streamlit web application** allows users to enter student details and instantly view predictions.

---

# Tech Stack

| Category         | Tools Used          |
| ---------------- | ------------------- |
| Programming      | Python              |
| Data Processing  | Pandas, NumPy       |
| Machine Learning | Scikit-learn        |
| Visualization    | Matplotlib, Seaborn |
| Web Interface    | Streamlit           |
| Model Storage    | Joblib              |

---

# Machine Learning Pipeline

The project follows a complete ML workflow:

1. **Data Collection**
2. **Data Cleaning & Preprocessing**
3. **Feature Engineering**
4. **Model Training**
5. **Hyperparameter Tuning**
6. **Model Evaluation**
7. **Deployment via Web App**

---

# Models Used

## Placement Prediction

**Model:** Random Forest Classifier

Key characteristics:

* Handles complex feature interactions
* Robust to overfitting
* Uses `class_weight="balanced"` to address class imbalance

**Hyperparameter Tuning:**
RandomizedSearchCV was used to optimize model performance.

---

## Salary Prediction

**Model:** Random Forest Regressor

Key characteristics:

* Captures nonlinear relationships
* Works well with tabular data
* Trained **only on students who were placed**

**Hyperparameter Tuning:**
RandomizedSearchCV

---

# Model Performance

## Placement Prediction

| Metric           | Score |
| ---------------- | ----- |
| Accuracy         | ~0.88 |
| F1 Score (Macro) | ~0.85 |

The balanced training improved the model's ability to correctly identify **non-placed students**.

---

## Salary Prediction

| Metric   | Score     |
| -------- | --------- |
| R² Score | ~0.75     |
| RMSE     | ~3.05 LPA |

The model explains a significant portion of salary variation while maintaining realistic error margins.

---

# Web Application

The project includes a **Streamlit-based interactive interface**.

Users can:

* Enter student details
* Predict placement outcome
* Estimate expected salary
* View feature importance affecting predictions

---

# Deployment

The application is deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project to a GitHub repository.
2. Go to **[https://share.streamlit.io](https://share.streamlit.io)**
3. Connect your repository.
4. Select `app.py` as the main file.
5. Deploy.

After deployment, the application becomes available at a public URL similar to:

```
https://your-app-name.streamlit.app
```

Every push to GitHub automatically updates the deployed app.

---

# Running the Project Locally

### 1. Clone the Repository

```
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Add Dataset

Place the following files inside the `data/` folder:

```
data/
 ├── students.csv
 └── placement.csv
```

### 4. Train the Models

```
python train_model.py
```

### 5. Run the Web Application

```
streamlit run app.py
```

---

# Using the Application

1. Enter student details such as:

   * Academic scores
   * Skills
   * Experience
   * Background information

2. Click:

```
Predict Placement & Salary
```

3. The app will display:

   * Placement prediction
   * Expected salary range
   * Important features influencing the prediction

---

# Project Structure

```
project/
│
├── data/
│   ├── students.csv
│   └── placement.csv
│
├── models/
│   ├── placement_model.pkl
│   └── salary_model.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

# Author

**Varun Shakya**
Machine Learning Student
