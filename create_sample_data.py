# Sample data structure for testing
# Replace this with your actual data files: students.csv and placement.csv

import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample data for testing the application"""

    # Sample students data
    students_data = {
        'Student_ID': range(1, 101),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'branch': np.random.choice(['Computer Science', 'Information Technology',
                                  'Electronics', 'Mechanical', 'Civil'], 100),
        'tenth_percentage': np.random.normal(85, 10, 100).clip(0, 100),
        'twelfth_percentage': np.random.normal(80, 12, 100).clip(0, 100),
        'cgpa': np.random.normal(7.5, 1.2, 100).clip(0, 10),
        'backlogs': np.random.poisson(0.5, 100),
        'projects_completed': np.random.poisson(2, 100),
        'internships_completed': np.random.poisson(1, 100),
        'hackathons_participated': np.random.poisson(0.8, 100),
        'coding_skill_rating': np.random.normal(6, 2, 100).clip(0, 10),
        'communication_skill_rating': np.random.normal(7, 1.5, 100).clip(0, 10),
        'aptitude_skill_rating': np.random.normal(6.5, 1.8, 100).clip(0, 10),
        'study_hours_per_day': np.random.normal(6, 2, 100).clip(0, 24),
        'attendance_percentage': np.random.normal(85, 10, 100).clip(0, 100),
        'part_time_job': np.random.choice(['Yes', 'No'], 100),
        'extracurricular_involvement': np.random.choice(['Low', 'Medium', 'High'], 100),
        'family_income_level': np.random.choice(['Low', 'Medium', 'High'], 100),
        'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], 100),
        'internet_access': np.random.choice(['Yes', 'No'], 100),
        'sleep_hours': np.random.normal(7, 1.5, 100).clip(0, 24),
        'stress_level': np.random.normal(5, 2, 100).clip(0, 10),
        'certifications_count': np.random.poisson(1.5, 100)
    }

    students_df = pd.DataFrame(students_data)

    # Sample placement data
    placement_data = {
        'Student_ID': range(1, 101),
        'placement_status': np.random.choice(['Placed', 'Not Placed'], 100, p=[0.7, 0.3]),
        'salary_lpa': np.random.normal(8, 3, 100).clip(0, 25)
    }

    placement_df = pd.DataFrame(placement_data)

    # Save to CSV
    students_df.to_csv('data/students.csv', index=False)
    placement_df.to_csv('data/placement.csv', index=False)

    print("Sample data created successfully!")
    print("Files saved to data/students.csv and data/placement.csv")

if __name__ == "__main__":
    create_sample_data()