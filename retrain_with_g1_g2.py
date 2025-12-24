"""
Retrain Student Performance Model with G1 and G2
File: retrain_with_g1_g2.py

Trains a new Random Forest model that uses G1 and G2 to predict G3
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import os

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('data/student_performance_full.csv')
print("✓ Data loaded:", df.shape)

# ============================================================================
# PREPARE DATA WITH G1 AND G2
# ============================================================================

# Remove G3 (target) from features
X = df.drop('G3', axis=1)
y = df['G3']

print("✓ Features shape:", X.shape)
print("✓ Target shape:", y.shape)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"✓ Categorical features: {len(categorical_cols)}")
print(f"✓ Numerical features: {len(numerical_cols)}")

# ============================================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================================

label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}")

# ============================================================================
# SPLIT DATA
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# ============================================================================
# SCALE FEATURES
# ============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")

# ============================================================================
# TRAIN MODEL WITH GRID SEARCH
# ============================================================================

print("\n🔍 Training Random Forest with Grid Search...")

param_grid = {
    'n_estimators': [50, 75, 100],
    'max_depth': [5, 7, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [3, 5]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best CV R² score: {grid_search.best_score_:.4f}")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

# Training metrics
y_train_pred = best_model.predict(X_train_scaled)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test metrics
y_test_pred = best_model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n📊 MODEL PERFORMANCE:")
print("=" * 50)
print(f"Training R² Score: {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print()
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print("=" * 50)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

feature_names = X_encoded.columns.tolist()
importances = best_model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

print("\n⭐ Top 10 Important Features:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i:2}. {feature_names[idx]:25} - {importances[idx]:.4f}")

# ============================================================================
# SAVE MODEL AND OBJECTS
# ============================================================================

os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(best_model, 'models/student_performance_model_with_g1_g2.pkl')
print("\n✓ Model saved: student_performance_model_with_g1_g2.pkl")

# Save scaler
joblib.dump(scaler, 'models/feature_scaler_with_g1_g2.pkl')
print("✓ Scaler saved: feature_scaler_with_g1_g2.pkl")

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders_with_g1_g2.pkl')
print("✓ Encoders saved: label_encoders_with_g1_g2.pkl")

# Save feature names
joblib.dump(feature_names, 'models/feature_names_with_g1_g2.pkl')
print("✓ Feature names saved: feature_names_with_g1_g2.pkl")

# Save summary
summary = {
    'model_type': 'Random Forest Regressor (with G1 & G2)',
    'optimization_method': 'GridSearchCV',
    'best_parameters': grid_search.best_params_,
    'training_r2_score': train_r2,
    'training_rmse': train_rmse,
    'training_mae': train_mae,
    'test_r2_score': test_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'num_features': len(feature_names),
    'feature_names': feature_names,
    'top_features': [(feature_names[i], float(importances[i])) for i in top_indices]
}

joblib.dump(summary, 'models/model_summary_with_g1_g2.pkl')
print("✓ Summary saved: model_summary_with_g1_g2.pkl")

print("\n" + "=" * 50)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 50)
print("\nNew model files created:")
print("  • student_performance_model_with_g1_g2.pkl")
print("  • feature_scaler_with_g1_g2.pkl")
print("  • label_encoders_with_g1_g2.pkl")
print("  • feature_names_with_g1_g2.pkl")
print("  • model_summary_with_g1_g2.pkl")
print("\nTo use these in app.py, update the file paths!")
