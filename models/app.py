"""
Flask Web App untuk Student Performance Prediction
File: app.py

Cara menjalankan:
    1. pip install flask scikit-learn pandas numpy joblib shap
    2. python app.py
    3. Buka http://localhost:5000 di browser
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import shap

app = Flask(__name__)

# ============================================================================
# LOAD MODEL & PREPROCESSING OBJECTS (saat startup)
# ============================================================================

MODEL_DIR = os.path.dirname(__file__)  # Direktori dimana model files berada

try:
    MODEL = joblib.load(os.path.join(MODEL_DIR, 'student_performance_model_with_g1_g2.pkl'))
    SCALER = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler_with_g1_g2.pkl'))
    LABEL_ENCODERS = joblib.load(os.path.join(MODEL_DIR, 'label_encoders_with_g1_g2.pkl'))
    FEATURE_NAMES = joblib.load(os.path.join(MODEL_DIR, 'feature_names_with_g1_g2.pkl'))
    SUMMARY = joblib.load(os.path.join(MODEL_DIR, 'model_summary_with_g1_g2.pkl'))
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ============================================================================
# INITIALIZE SHAP EXPLAINER (for feature contribution analysis)
# ============================================================================

try:
    # Create a SHAP TreeExplainer for the Random Forest model
    # Note: This is disabled for now as it's very slow to initialize
    # Instead, we'll use permutation-based feature importance during prediction
    EXPLAINER = None  # shap.TreeExplainer(MODEL)
    print("SHAP support loaded (lazy initialization)")
except Exception as e:
    print(f"Warning: SHAP Explainer initialization failed: {e}")
    EXPLAINER = None

# ============================================================================
# CONFIGURATION
# ============================================================================

CATEGORICAL_FEATURES = list(LABEL_ENCODERS.keys())
NUMERICAL_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]

# Categorical feature options (sesuaikan dengan training data)
FEATURE_OPTIONS = {
    'school': ['GP', 'MS'],
    'sex': ['F', 'M'],
    'address': ['R', 'U'],
    'famsize': ['GT3', 'LE3'],
    'pstatus': ['A', 'T'],
    'mjob': ['at_home', 'health', 'other', 'services', 'teacher'],
    'fjob': ['at_home', 'health', 'other', 'services', 'teacher'],
    'reason': ['course', 'home', 'other', 'reputation'],
    'guardian': ['father', 'mother', 'other'],
    'schoolsup': ['no', 'yes'],
    'famsup': ['no', 'yes'],
    'paid': ['no', 'yes'],
    'activities': ['no', 'yes'],
    'nursery': ['no', 'yes'],
    'higher': ['no', 'yes'],
    'internet': ['no', 'yes'],
    'romantic': ['no', 'yes']
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_shap_values(X_scaled, X_original, feature_names):
    """
    Compute feature contributions berdasarkan model predictions
    Menggunakan permutation importance approach yang lebih ringan
    
    Args:
        X_scaled: Scaled feature array untuk prediction
        X_original: Original feature values
        feature_names: List of feature names
        
    Returns:
        dict: Feature contributions dengan kontribusi setiap fitur
    """
    try:
        # Get base prediction
        base_pred = MODEL.predict(X_scaled)[0]
        
        # Compute feature contributions by measuring impact of each feature
        contributions = []
        for i, feature_name in enumerate(feature_names):
            # Get feature value
            feature_value = X_original[feature_name] if feature_name in X_original else 0
            
            # Create a copy and zero out this feature
            X_permuted = X_scaled.copy()
            X_permuted[0, i] = 0  # Zero out the feature
            
            # Get prediction with permuted feature
            permuted_pred = MODEL.predict(X_permuted)[0]
            
            # Contribution is the difference
            contribution = base_pred - permuted_pred
            
            contributions.append({
                'feature': feature_name,
                'contribution': round(float(contribution), 4),
                'abs_contribution': abs(round(float(contribution), 4))
            })
        
        # Sort by absolute contribution
        contributions_sorted = sorted(contributions, key=lambda x: x['abs_contribution'], reverse=True)
        
        return {
            'base_value': float(base_pred),
            'contributions': contributions_sorted[:10]  # Top 10
        }
    except Exception as e:
        print(f"Contribution computation error: {e}")
        return None

def preprocess_input(input_data):
    """
    Preprocess user input sebelum prediction
    
    Args:
        input_data (dict): Dictionary dengan feature values
        
    Returns:
        array: Preprocessed data siap untuk prediction
    """
    
    try:
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # All features used in training
        scaler_features = FEATURE_NAMES
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in scaler_features:
                le = LABEL_ENCODERS[col]
                df[col] = le.transform([str(df[col].values[0])])[0]
        
        # Select only features that were used in scaler fit and scale them
        df_for_scaling = df[scaler_features]
        df_scaled = SCALER.transform(df_for_scaling)
        
        return df_scaled, None
    
    except Exception as e:
        return None, str(e)


def classify_performance(grade):
    """Classify student performance based on grade"""
    if grade >= 15:
        return 'Sangat Baik', 'success'
    elif grade >= 12:
        return 'Baik', 'info'
    elif grade >= 10:
        return 'Cukup', 'warning'
    else:
        return 'Perlu Perbaikan', 'danger'


def get_top_features():
    """Get top 10 important features"""
    importances = MODEL.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]
    top_features = [(FEATURE_NAMES[i], round(importances[i], 4)) for i in indices]
    return top_features


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Homepage - prediction form"""
    return render_template('index.html', features=FEATURE_OPTIONS)


@app.route('/simple')
def simple():
    """Simplified prediction form with only 10 key features"""
    return render_template('simple.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint untuk prediction
    
    Expected JSON:
    {
        "school": "GP",
        "sex": "M",
        "age": 18,
        ... (semua 32 features: all except G3)
    }
    """
    try:
        data = request.get_json()
        
        # Define actual training features (all 32 features)
        required_features = FEATURE_NAMES
        
        # Validate required features
        missing = [f for f in required_features if f not in data or data[f] == '']
        if missing:
            return jsonify({
                'error': f'Missing features: {", ".join(missing[:5])}'
            }), 400
        
        # Convert to correct types
        for col in data:
            if col in NUMERICAL_FEATURES:
                data[col] = float(data[col])
            else:
                data[col] = str(data[col])
        
        # Preprocess
        X_processed, error = preprocess_input(data)
        if error:
            return jsonify({'error': f'Preprocessing error: {error}'}), 400
        
        # Predict
        prediction = MODEL.predict(X_processed)[0]
        status, status_class = classify_performance(prediction)
        
        # Get feature importance dan contributions
        top_features = get_top_features()
        shap_result = compute_shap_values(X_processed, data, FEATURE_NAMES)
        
        # Format SHAP contributions untuk response
        shap_contributions = None
        if shap_result:
            shap_contributions = [
                {
                    'feature': c['feature'],
                    'contribution': c['contribution'],
                    'contribution_pct': round((c['abs_contribution'] / max([x['abs_contribution'] for x in shap_result['contributions']] + [0.0001])) * 100, 2)
                }
                for c in shap_result['contributions']
            ]
        
        return jsonify({
            'success': True,
            'predicted_grade': round(float(prediction), 2),
            'grade_range': 'G3 (0-20)',
            'status': status,
            'status_class': status_class,
            'model_info': {
                'type': SUMMARY['model_type'],
                'test_r2': round(SUMMARY['test_r2_score'], 4),
                'test_rmse': round(SUMMARY['test_rmse'], 4),
                'test_mae': round(SUMMARY['test_mae'], 4)
            },
            'top_features': top_features,
            'shap_contributions': shap_contributions
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Get model information"""
    return jsonify({
        'model_type': SUMMARY['model_type'],
        'optimization': SUMMARY['optimization_method'],
        'best_parameters': SUMMARY['best_parameters'],
        'performance': {
            'test_r2': round(SUMMARY['test_r2_score'], 4),
            'test_rmse': round(SUMMARY['test_rmse'], 4),
            'test_mae': round(SUMMARY['test_mae'], 4)
        },
        'dataset': {
            'total_features': SUMMARY['num_features']
        },
        'features': {
            'categorical': CATEGORICAL_FEATURES,
            'numerical': NUMERICAL_FEATURES
        }
    })


@app.route('/api/features', methods=['GET'])
def api_features():
    """Get feature options untuk form"""
    return jsonify(FEATURE_OPTIONS)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STUDENT PERFORMANCE PREDICTION WEB APP")
    print("="*80)
    print(f"\n✓ Model: {SUMMARY['model_type']}")
    print(f"✓ Parameters: {SUMMARY['best_parameters']}")
    print(f"✓ Features: {len(FEATURE_NAMES)}")
    print(f"\n🚀 Starting server at http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
