# Student Performance Prediction Models

## 📋 Overview

Folder ini berisi semua model dan preprocessing objects yang sudah di-train dan di-export untuk digunakan dalam web application.

## 📁 File-File yang Tersedia

### Model Files
- **`student_performance_model_optimized.pkl`** - Model Random Forest yang sudah di-optimize dengan GridSearchCV (RECOMMENDED)
- **`student_performance_model_original.pkl`** - Model Random Forest original (untuk comparison)
- **`tpot_best_pipeline.py`** - Pipeline TPOT dalam format Python (dokumentasi)
- **`tpot_model.pkl`** - Model TPOT yang sudah di-train (backup)

### Preprocessing Files
- **`feature_scaler.pkl`** - StandardScaler untuk normalisasi feature
- **`label_encoders.pkl`** - Dictionary berisi LabelEncoder untuk categorical features (12 features)
- **`feature_names.pkl`** - List nama-nama feature dalam urutan yang benar
- **`feature_info.pkl`** - Info lengkap tentang numerical & categorical features

### Documentation
- **`model_summary_optimized.pkl`** - Metadata lengkap model (parameters, performance, dll)
- **`tpot_best_pipeline_info.txt`** - Info detail tentang TPOT pipeline
- **`tpot_best_pipeline.py`** - Template Python untuk menggunakan pipeline TPOT

---

## 🚀 Cara Menggunakan di Web App

### Step 1: Load Model & Preprocessing Objects

```python
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('student_performance_model_optimized.pkl')

# Load preprocessing objects
scaler = joblib.load('feature_scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')
feature_info = joblib.load('feature_info.pkl')

# Load model summary untuk info
summary = joblib.load('model_summary_optimized.pkl')
print(f"Model: {summary['model_type']}")
print(f"Parameters: {summary['best_parameters']}")
```

### Step 2: Prepare Input Data

```python
def preprocess_input(input_data):
    """
    Preprocess user input sebelum prediction
    
    Args:
        input_data (dict): Dictionary dengan feature values dari user
        
    Returns:
        array: Preprocessed data siap untuk prediction
    """
    
    # Create DataFrame dari input
    df = pd.DataFrame([input_data])
    
    # 1. Encode categorical features
    categorical_features = list(label_encoders.keys())
    for col in categorical_features:
        if col in df.columns:
            # Use the label encoder untuk encode value
            le = label_encoders[col]
            df[col] = le.transform([df[col].values[0]])[0]
    
    # 2. Pastikan urutan column sesuai dengan training data
    df = df[feature_names]
    
    # 3. Scale numerical features
    df_scaled = scaler.transform(df)
    
    return df_scaled
```

### Step 3: Make Prediction

```python
def predict_student_grade(input_data):
    """
    Predict student final grade
    
    Args:
        input_data (dict): Dictionary dengan student features
                          Example: {
                              'school': 'GP',
                              'sex': 'M',
                              'age': 18,
                              'address': 'U',
                              ... (semua feature)
                          }
        
    Returns:
        dict: Prediction result dengan grade dan confidence
    """
    
    try:
        # Preprocess input
        X_processed = preprocess_input(input_data)
        
        # Make prediction
        y_pred = model.predict(X_processed)
        predicted_grade = y_pred[0]
        
        # Get feature importances untuk explanation
        feature_importance = model.feature_importances_
        
        # Create result
        result = {
            'predicted_grade': round(predicted_grade, 2),
            'status': classify_performance(predicted_grade),
            'model_type': 'Random Forest (Optimized)',
            'confidence': get_confidence_score(feature_importance)
        }
        
        return result
    
    except Exception as e:
        return {'error': str(e)}


def classify_performance(grade):
    """Classify grade range"""
    if grade >= 15:
        return 'Excellent'
    elif grade >= 12:
        return 'Good'
    elif grade >= 10:
        return 'Satisfactory'
    else:
        return 'Needs Improvement'


def get_confidence_score(importance):
    """Calculate confidence dari model"""
    # Semakin tinggi max importance, semakin confident
    confidence = np.max(importance) * 100
    return min(round(confidence, 1), 100)
```

---

## 💻 Contoh Flask Web App

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model once at startup
MODEL = joblib.load('student_performance_model_optimized.pkl')
SCALER = joblib.load('feature_scaler.pkl')
LABEL_ENCODERS = joblib.load('label_encoders.pkl')
FEATURE_NAMES = joblib.load('feature_names.pkl')
SUMMARY = joblib.load('model_summary_optimized.pkl')

# Define all feature names and types
CATEGORICAL_FEATURES = list(LABEL_ENCODERS.keys())
NUMERICAL_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint untuk prediction
    
    Expected JSON input:
    {
        "school": "GP",
        "sex": "M",
        "age": 18,
        "address": "U",
        "famsize": "GT3",
        ... (semua 30 features)
    }
    """
    try:
        data = request.get_json()
        
        # Validation
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            le = LABEL_ENCODERS[col]
            df[col] = le.transform([df[col].values[0]])[0]
        
        # Ensure correct column order
        df = df[FEATURE_NAMES]
        
        # Scale features
        df_scaled = SCALER.transform(df)
        
        # Predict
        prediction = MODEL.predict(df_scaled)[0]
        
        # Response
        return jsonify({
            'predicted_grade': round(float(prediction), 2),
            'status': classify_performance(prediction),
            'model_info': {
                'type': SUMMARY['model_type'],
                'parameters': SUMMARY['best_parameters'],
                'test_r2': SUMMARY['optimized_test_r2_score'],
                'test_rmse': SUMMARY['optimized_test_rmse']
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information dan feature list"""
    return jsonify({
        'model_type': SUMMARY['model_type'],
        'parameters': SUMMARY['best_parameters'],
        'performance': {
            'test_r2': SUMMARY['optimized_test_r2_score'],
            'test_rmse': SUMMARY['optimized_test_rmse'],
            'test_mae': SUMMARY['optimized_test_mae'],
            'overfitting_reduction': SUMMARY['overfitting_reduction']
        },
        'features': {
            'total': len(FEATURE_NAMES),
            'categorical': CATEGORICAL_FEATURES,
            'numerical': NUMERICAL_FEATURES
        }
    })


def classify_performance(grade):
    if grade >= 15:
        return 'Excellent'
    elif grade >= 12:
        return 'Good'
    elif grade >= 10:
        return 'Satisfactory'
    else:
        return 'Needs Improvement'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Cara Menggunakan Flask App:

```bash
# Install dependencies
pip install flask scikit-learn pandas numpy joblib

# Run app
python app.py

# Test prediction (curl)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP",
    "sex": "M",
    "age": 18,
    "address": "U",
    "famsize": "GT3",
    ... (semua 30 features)
  }'

# Get model info
curl http://localhost:5000/model-info
```

---

## 🔧 FastAPI Alternative (Modern)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Dict, List

app = FastAPI()

# Load models
MODEL = joblib.load('student_performance_model_optimized.pkl')
SCALER = joblib.load('feature_scaler.pkl')
LABEL_ENCODERS = joblib.load('label_encoders.pkl')
FEATURE_NAMES = joblib.load('feature_names.pkl')
SUMMARY = joblib.load('model_summary_optimized.pkl')


class StudentInput(BaseModel):
    """Input schema untuk student features"""
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    pstatus: str
    medu: int
    fedu: int
    mjob: str
    fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    dalc: int
    walc: int
    health: int
    absences: int


class PredictionResponse(BaseModel):
    predicted_grade: float
    status: str
    model_type: str


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    try:
        # Convert to dict
        data = student.dict()
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical
        categorical_features = list(LABEL_ENCODERS.keys())
        for col in categorical_features:
            le = LABEL_ENCODERS[col]
            df[col] = le.transform([df[col].values[0]])[0]
        
        # Order columns
        df = df[FEATURE_NAMES]
        
        # Scale
        df_scaled = SCALER.transform(df)
        
        # Predict
        prediction = MODEL.predict(df_scaled)[0]
        
        return PredictionResponse(
            predicted_grade=round(float(prediction), 2),
            status=classify_performance(prediction),
            model_type=SUMMARY['model_type']
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info")
def get_model_info():
    return {
        'model_type': SUMMARY['model_type'],
        'parameters': SUMMARY['best_parameters'],
        'performance': {
            'test_r2': SUMMARY['optimized_test_r2_score'],
            'test_rmse': SUMMARY['optimized_test_rmse']
        }
    }


def classify_performance(grade):
    if grade >= 15:
        return 'Excellent'
    elif grade >= 12:
        return 'Good'
    elif grade >= 10:
        return 'Satisfactory'
    else:
        return 'Needs Improvement'
```

### Jalankan FastAPI:

```bash
# Install
pip install fastapi uvicorn

# Run
uvicorn app:app --reload

# Visit docs
http://localhost:8000/docs
```

---

## 📊 Model Performance Summary

| Metric | Value |
|--------|-------|
| Model Type | Random Forest (Optimized) |
| Optimization | GridSearchCV with 5-fold CV |
| Test R² Score | 0.2079 |
| Test RMSE | 2.7793 |
| Test MAE | 2.0413 |
| Overfitting Reduction | 43.47% |

---

## ✅ Checklist Sebelum Deploy

- [ ] Load semua model files dengan benar
- [ ] Validate input sesuai dengan feature names
- [ ] Scale input dengan `feature_scaler.pkl` yang sama
- [ ] Encode categorical features dengan `label_encoders.pkl` yang benar
- [ ] Maintain order feature names sesuai `feature_names.pkl`
- [ ] Implement error handling untuk invalid input
- [ ] Add logging untuk tracking predictions
- [ ] Test dengan sample data sebelum production
- [ ] Setup CORS jika API akan diakses dari frontend domain berbeda

---

## 📝 Important Notes

1. **Feature Order Matters**: Selalu gunakan urutan feature dari `feature_names.pkl`
2. **Encoding Consistency**: Gunakan `label_encoders.pkl` yang sama untuk semua prediction
3. **Scaling**: WAJIB scale input dengan `feature_scaler.pkl` yang digunakan saat training
4. **Categorical Features**: 12 features categorical harus di-encode dengan label encoder
5. **Grade Range**: Output grade biasanya range 0-20 (skala Portugis)

---

## 🔗 Feature Explanation

| Feature | Type | Description |
|---------|------|-------------|
| school | Categorical | School type (GP, MS) |
| sex | Categorical | Student gender |
| age | Numerical | Student age |
| address | Categorical | Address type (U=urban, R=rural) |
| famsize | Categorical | Family size |
| pstatus | Categorical | Parent's cohabitation status |
| medu | Numerical | Mother's education |
| fedu | Numerical | Father's education |
| ... | ... | ... (30 features total) |

Untuk detail lengkap semua features, lihat notebook documentation.
