import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load models and feature names
@st.cache_resource
def load_models():
    try:
        clf_model = joblib.load('models/placement_classifier.pkl')
        reg_model = joblib.load('models/salary_regressor.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return clf_model, reg_model, feature_names
    except FileNotFoundError:
        st.error("Models not found. Please train the models first by running train_model.py")
        return None, None, None

def main():
    st.title("🎓 Student Placement & Salary Prediction")
    st.markdown("Predict student placement status and expected salary based on academic and skill parameters.")

    # Load models
    clf_model, reg_model, feature_names = load_models()

    if clf_model is None:
        return

    # determine available branch columns from saved features
    branch_cols = [col for col in feature_names if col.startswith('branch_')]
    # derive branch names for UI
    branch_options = [col.replace('branch_', '') for col in branch_cols]

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Academic Information")
        twelfth_percentage = st.slider("12th Percentage", 0.0, 100.0, 85.0)
        cgpa = st.slider("CGPA", 0.0, 10.0, 8.0)
        backlogs = st.number_input("Backlogs", 0, 10, 0)

    with col2:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", branch_options)
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Skills & Experience")
        coding_skill = st.slider("Coding Skill Rating", 0.0, 10.0, 5.0)
        communication_skill = st.slider("Communication Skill Rating", 0.0, 10.0, 6.0)
        aptitude_skill = st.slider("Aptitude Skill Rating", 0.0, 10.0, 6.0)

        projects_completed = st.number_input("Projects Completed", 0, 20, 2)
        internships_completed = st.number_input("Internships Completed", 0, 10, 1)
        hackathons_participated = st.number_input("Hackathons Participated", 0, 10, 0)

        extracurricular = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])

    with col4:
        st.subheader("Other Factors")
        part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])

    # Prediction button
    if st.button("Predict Placement & Salary", type="primary"):
        # Prepare input data
        # compute combined features from individual inputs
        practical_experience = projects_completed + internships_completed + hackathons_participated
        skill_rating = (0.5 * coding_skill) + (0.2 * communication_skill) + (0.3 * aptitude_skill)

        input_data = {
            'twelfth_percentage': twelfth_percentage,
            'cgpa': cgpa,
            'backlogs': backlogs,
            'gender': 1 if gender == "Male" else 0,
            'city_tier': {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}[city_tier],
            'family_income_level': {"Low": 1, "Medium": 2, "High": 3}[family_income],
            'practical_experience': practical_experience,
            'skill_rating': skill_rating,
            'extracurricular_involvement': {"Low": 1, "Medium": 2, "High": 3}[extracurricular],
            'part_time_job': 1 if part_time_job == "Yes" else 0,
            'internet_access': 1 if internet_access == "Yes" else 0
        }

        # Add branch one-hot encoding using the same columns used during training
        for col in branch_cols:
            input_data[col] = 1 if col == f"branch_{branch}" else 0

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]  # Ensure correct column order

        # Make predictions
        placement_prob = clf_model.predict_proba(input_df)[0][1]
        placement_pred = clf_model.predict(input_df)[0]

        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if placement_pred == 1:
            st.success(f"🎉 **Placed!** (Confidence: {placement_prob:.1%})")

            # Salary prediction
            salary_pred = reg_model.predict(input_df)[0]
            # show the predicted salary
            st.info(f"Predicted salary: {salary_pred:.1f} LPA")
            # Display salary range
            salary_range = f"₹{(salary_pred - 2):.1f} - ₹{(salary_pred + 2):.1f} LPA"
            st.metric("Expected Salary Range", salary_range)
        else:
            st.error(f"❌ **Not Placed** (Confidence: {(1-placement_prob):.1%})")
            st.info("💡 **Tips for Improvement:** Focus on improving CGPA, gaining practical experience, and developing technical skills.")

        # Feature importance visualization
        st.markdown("---")
        st.subheader("🔍 Key Factors Analysis")

        # Get feature importances
        clf_importance = clf_model.feature_importances_
        reg_importance = reg_model.feature_importances_

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Placement_Importance': clf_importance,
            'Salary_Importance': reg_importance
        })

        # Display top factors
        col5, col6 = st.columns(2)

        with col5:
            st.markdown("**Top factors for Placement:**")
            top_placement = importance_df.nlargest(5, 'Placement_Importance')
            for _, row in top_placement.iterrows():
                st.write(f"• {row['Feature'].replace('_', ' ').title()}: {row['Placement_Importance']:.3f}")

        with col6:
            st.markdown("**Top factors for Salary:**")
            top_salary = importance_df.nlargest(5, 'Salary_Importance')
            for _, row in top_salary.iterrows():
                st.write(f"• {row['Feature'].replace('_', ' ').title()}: {row['Salary_Importance']:.3f}")

if __name__ == "__main__":
    main()