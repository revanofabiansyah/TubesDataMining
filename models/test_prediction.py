"""
Simple Python Script untuk Test Prediction
File: test_prediction.py

Cara menggunakan:
    python test_prediction.py
"""

import joblib
import pandas as pd
import numpy as np
import os

# ============================================================================
# LOAD MODEL & PREPROCESSING
# ============================================================================

MODEL_DIR = os.path.dirname(__file__)

print("Loading model and preprocessing objects...")
model = joblib.load(os.path.join(MODEL_DIR, 'student_performance_model_optimized.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
summary = joblib.load(os.path.join(MODEL_DIR, 'model_summary_optimized.pkl'))

print("✓ Model loaded successfully!\n")

# ============================================================================
# MODEL INFO
# ============================================================================

print("=" * 80)
print("MODEL INFORMATION")
print("=" * 80)
print(f"Model Type: {summary['model_type']}")
print(f"Best Parameters: {summary['best_parameters']}")
print(f"Test R² Score: {summary['optimized_test_r2_score']:.4f}")
print(f"Test RMSE: {summary['optimized_test_rmse']:.4f}")
print(f"Test MAE: {summary['optimized_test_mae']:.4f}")
print()

# ============================================================================
# SAMPLE DATA FOR PREDICTION
# ============================================================================

# Example 1: Good student
sample_data_1 = {
    'school': 'GP',
    'sex': 'F',
    'age': 18,
    'address': 'U',
    'famsize': 'GT3',
    'pstatus': 'T',
    'medu': 4,
    'fedu': 4,
    'mjob': 'teacher',
    'fjob': 'teacher',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 1,
    'studytime': 3,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'yes',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'no',
    'famrel': 5,
    'freetime': 3,
    'goout': 2,
    'dalc': 1,
    'walc': 1,
    'health': 5,
    'absences': 2
}

# Example 2: Average student
sample_data_2 = {
    'school': 'GP',
    'sex': 'M',
    'age': 18,
    'address': 'R',
    'famsize': 'LE3',
    'pstatus': 'T',
    'medu': 2,
    'fedu': 2,
    'mjob': 'services',
    'fjob': 'services',
    'reason': 'home',
    'guardian': 'mother',
    'traveltime': 2,
    'studytime': 2,
    'failures': 1,
    'schoolsup': 'no',
    'famsup': 'no',
    'paid': 'no',
    'activities': 'no',
    'nursery': 'no',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'yes',
    'famrel': 3,
    'freetime': 3,
    'goout': 3,
    'dalc': 2,
    'walc': 2,
    'health': 3,
    'absences': 5
}

# Example 3: Struggling student
sample_data_3 = {
    'school': 'MS',
    'sex': 'M',
    'age': 19,
    'address': 'R',
    'famsize': 'LE3',
    'pstatus': 'A',
    'medu': 1,
    'fedu': 1,
    'mjob': 'at_home',
    'fjob': 'other',
    'reason': 'home',
    'guardian': 'father',
    'traveltime': 4,
    'studytime': 1,
    'failures': 3,
    'schoolsup': 'yes',
    'famsup': 'no',
    'paid': 'no',
    'activities': 'no',
    'nursery': 'no',
    'higher': 'no',
    'internet': 'no',
    'romantic': 'yes',
    'famrel': 2,
    'freetime': 4,
    'goout': 4,
    'dalc': 4,
    'walc': 4,
    'health': 2,
    'absences': 20
}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_grade(student_data, student_name="Student"):
    """
    Make prediction for a student
    
    Args:
        student_data (dict): Dictionary dengan feature values
        student_name (str): Nama student untuk display
        
    Returns:
        tuple: (predicted_grade, status, status_class)
    """
    
    try:
        # Create DataFrame
        df = pd.DataFrame([student_data])
        
        # Encode categorical features
        categorical_features = list(label_encoders.keys())
        for col in categorical_features:
            if col in df.columns:
                le = label_encoders[col]
                df[col] = le.transform([str(df[col].values[0])])[0]
        
        # Ensure correct column order
        df = df[feature_names]
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        
        # Classify
        if prediction >= 15:
            status = 'Excellent'
            status_class = '🌟'
        elif prediction >= 12:
            status = 'Good'
            status_class = '😊'
        elif prediction >= 10:
            status = 'Satisfactory'
            status_class = '👍'
        else:
            status = 'Needs Improvement'
            status_class = '⚠️'
        
        return prediction, status, status_class
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

print("=" * 80)
print("PREDICTION EXAMPLES")
print("=" * 80)

samples = [
    (sample_data_1, "Good Student (High achiever)"),
    (sample_data_2, "Average Student (Normal performance)"),
    (sample_data_3, "Struggling Student (At-risk)")
]

results = []

for sample_data, description in samples:
    print(f"\n{description}")
    print("-" * 80)
    
    # Get important features
    print(f"Student Profile:")
    print(f"  • School: {sample_data['school']}")
    print(f"  • Gender: {'Female' if sample_data['sex'] == 'F' else 'Male'}")
    print(f"  • Age: {sample_data['age']}")
    print(f"  • Address: {'Urban' if sample_data['address'] == 'U' else 'Rural'}")
    print(f"  • Study Time: {sample_data['studytime']}/4")
    print(f"  • Failures: {sample_data['failures']}")
    print(f"  • Family Support: {'Yes' if sample_data['famsup'] == 'yes' else 'No'}")
    
    # Predict
    grade, status, status_class = predict_grade(sample_data, description)
    
    if grade is not None:
        print(f"\nPrediction Results:")
        print(f"  • Predicted Grade: {grade:.2f}/20")
        print(f"  • Status: {status_class} {status}")
        
        # Store result
        results.append({
            'description': description,
            'grade': grade,
            'status': status
        })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for result in results:
    print(f"{result['description']}: {result['grade']:.2f} ({result['status']})")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 80)

importances = model.feature_importances_
indices = np.argsort(importances)[-10:][::-1]

for i, idx in enumerate(indices, 1):
    feature = feature_names[idx]
    importance = importances[idx]
    bar_length = int(importance * 50)
    bar = '█' * bar_length
    print(f"{i:2d}. {feature:20s} {bar} {importance:.4f}")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\n" + "=" * 80)
print("INTERACTIVE MODE")
print("=" * 80)
print("\nWould you like to make a custom prediction? (y/n): ", end="")

user_input = input().strip().lower()

if user_input == 'y':
    print("\nEnter student data (or press Enter to use default values):\n")
    
    custom_data = sample_data_1.copy()  # Start with default
    
    # Ask for key features
    print("Key Features:")
    
    # Age
    age_input = input("Age (15-25) [18]: ").strip()
    if age_input:
        custom_data['age'] = int(age_input)
    
    # Study time
    study_input = input("Study Time (1-4) [3]: ").strip()
    if study_input:
        custom_data['studytime'] = int(study_input)
    
    # Failures
    fail_input = input("Number of Failures (0-3) [0]: ").strip()
    if fail_input:
        custom_data['failures'] = int(fail_input)
    
    # Family support
    famsup_input = input("Family Support (yes/no) [yes]: ").strip().lower()
    if famsup_input in ['yes', 'no']:
        custom_data['famsup'] = famsup_input
    
    # Predict
    print("\n" + "=" * 80)
    grade, status, status_class = predict_grade(custom_data, "Custom Student")
    
    if grade is not None:
        print(f"\nCustom Prediction Result:")
        print(f"  • Predicted Grade: {grade:.2f}/20")
        print(f"  • Status: {status_class} {status}")

print("\n✨ Test complete! Ready for production deployment.\n")
