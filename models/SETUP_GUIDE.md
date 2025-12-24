# SETUP GUIDE: Student Performance Prediction Web App

## 📋 Quick Start

### 1. Install Dependencies

```bash
# Pastikan berada di folder models
cd c:\Users\REVANO PC\Documents\TubesDataMining\models

# Install required packages
pip install -r requirements.txt
```

### 2. Run Flask App

```bash
# Jalankan aplikasi
python app.py
```

Aplikasi akan berjalan di: **http://localhost:5000**

---

## 📁 Struktur Folder

```
models/
├── app.py                          # Flask application main
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── SETUP_GUIDE.md                 # File ini
│
├── templates/
│   └── index.html                  # Web UI
│
├── student_performance_model_optimized.pkl    # ⭐ Main model
├── student_performance_model_original.pkl     # Backup model
├── feature_scaler.pkl              # Feature normalization
├── label_encoders.pkl              # Categorical encoding
├── feature_names.pkl               # Feature order
├── feature_info.pkl                # Feature metadata
├── model_summary_optimized.pkl     # Model info
│
├── tpot_best_pipeline.py           # TPOT pipeline code
├── tpot_best_pipeline_info.txt     # TPOT details
└── tpot_model.pkl                  # TPOT model backup
```

---

## 🚀 Cara Menggunakan

### A. Web Interface (GUI)

1. **Start aplikasi:**
   ```bash
   python app.py
   ```

2. **Buka browser:**
   ```
   http://localhost:5000
   ```

3. **Isi formulir dengan data student:**
   - Personal info (name, age, gender, dll)
   - Parents education & job
   - Academic info (study time, failures, dll)
   - Personal habits (family relations, health, dll)

4. **Klik "Predict Grade"**

5. **Hasil akan ditampilkan dengan:**
   - Predicted grade (0-20)
   - Performance status (Excellent/Good/Satisfactory/Needs Improvement)
   - Top important features
   - Model metrics

### B. API Endpoints (untuk Development)

#### 1. Prediction Endpoint

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "GP",
    "sex": "M",
    "age": 18,
    "address": "U",
    "famsize": "GT3",
    "pstatus": "T",
    "medu": 3,
    "fedu": 3,
    "mjob": "teacher",
    "fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "yes",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "dalc": 1,
    "walc": 1,
    "health": 5,
    "absences": 2
  }'
```

**Response (Success):**
```json
{
  "success": true,
  "predicted_grade": 14.52,
  "grade_range": "G3 (0-20)",
  "status": "Good",
  "status_class": "info",
  "model_info": {
    "type": "Random Forest (Optimized)",
    "test_r2": 0.2079,
    "test_rmse": 2.7793,
    "test_mae": 2.0413
  },
  "top_features": [
    ["G2", 0.4523],
    ["studytime", 0.1234],
    ...
  ]
}
```

#### 2. Model Info Endpoint

```bash
curl http://localhost:5000/api/model-info
```

**Response:**
```json
{
  "model_type": "Random Forest (Optimized)",
  "optimization": "GridSearchCV with 5-fold CV",
  "best_parameters": {
    "max_depth": 7,
    "min_samples_leaf": 5,
    "min_samples_split": 20,
    "n_estimators": 75
  },
  "performance": {
    "test_r2": 0.2079,
    "test_rmse": 2.7793,
    "test_mae": 2.0413,
    "overfitting_reduction": "43.47%"
  },
  ...
}
```

#### 3. Features Endpoint

```bash
curl http://localhost:5000/api/features
```

Returns list of categorical feature options.

---

## 🔧 Python Integration (Non-Web)

Jika ingin menggunakan model langsung di Python tanpa web:

```python
import joblib
import pandas as pd
import numpy as np

# Load model & preprocessing
model = joblib.load('student_performance_model_optimized.pkl')
scaler = joblib.load('feature_scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

# Prepare data
data = {
    'school': 'GP',
    'sex': 'M',
    'age': 18,
    # ... (semua 30 features)
}

# Create DataFrame
df = pd.DataFrame([data])

# Encode categorical
for col, le in label_encoders.items():
    df[col] = le.transform([df[col].values[0]])[0]

# Order & scale
df = df[feature_names]
df_scaled = scaler.transform(df)

# Predict
grade = model.predict(df_scaled)[0]
print(f"Predicted Grade: {grade:.2f}")
```

---

## 📊 Model Information

| Aspect | Value |
|--------|-------|
| Model | Random Forest Regressor |
| Optimization | GridSearchCV (5-fold CV) |
| Best Parameters | n_estimators: 75, max_depth: 7, min_samples_leaf: 5, min_samples_split: 20 |
| Training Samples | 519 |
| Test Samples | 130 |
| Features | 32 (30 original + 2 engineered) |
| Test R² Score | 0.2079 |
| Test RMSE | 2.7793 |
| Test MAE | 2.0413 |
| Overfitting Reduction | 43.47% |

---

## ✅ Troubleshooting

### Problem 1: "ModuleNotFoundError: No module named 'joblib'"
**Solution:**
```bash
pip install joblib scikit-learn pandas numpy
```

### Problem 2: "FileNotFoundError: [Errno 2] No such file or directory: 'student_performance_model_optimized.pkl'"
**Solution:**
- Pastikan semua .pkl files ada di folder models
- Jalankan app.py dari folder models
- Atau update path di app.py

### Problem 3: "Port 5000 already in use"
**Solution:**
```bash
# Gunakan port lain
python -c "app.run(port=5001)"
```

### Problem 4: Form validation error
**Solution:**
- Pastikan semua field terisi
- Pastikan nilai numerik sesuai dengan range:
  - Age: 15-25
  - Education: 0-4
  - Health/Relations: 1-5
  - Absences: 0-93

---

## 🌐 Production Deployment

### Option 1: Gunicorn (Linux/Mac)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 2: Docker

Buat `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

Run Docker:
```bash
docker build -t student-prediction .
docker run -p 5000:5000 student-prediction
```

### Option 3: Heroku

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Deploy:
```bash
git push heroku main
```

---

## 📝 API Usage Examples

### JavaScript/Frontend

```javascript
async function predictGrade(studentData) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(studentData)
    });
    
    const result = await response.json();
    if (result.success) {
        console.log(`Grade: ${result.predicted_grade}`);
        console.log(`Status: ${result.status}`);
    }
}
```

### Python Requests

```python
import requests
import json

data = {
    'school': 'GP',
    'sex': 'M',
    # ... (30 features)
}

response = requests.post(
    'http://localhost:5000/api/predict',
    json=data
)

result = response.json()
print(f"Predicted Grade: {result['predicted_grade']}")
```

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Run Flask app
3. ✅ Test prediction di web UI
4. ✅ Integrate dengan frontend/mobile app jika perlu
5. ✅ Deploy ke production

---

## 📞 Support

Jika ada masalah:
1. Check error message di console
2. Verify semua .pkl files ada
3. Check Python version (3.8+)
4. Reinstall dependencies

---

**Happy Predicting! 🎓**
