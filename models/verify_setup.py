#!/usr/bin/env python
"""Quick test to verify all components are working"""

import joblib
import sys

print("=" * 80)
print("VERIFICATION TEST")
print("=" * 80)

try:
    print("\n✓ Loading model...")
    model = joblib.load('student_performance_model_optimized.pkl')
    print(f"  ✅ Model loaded: {type(model).__name__}")
    
    print("\n✓ Loading scaler...")
    scaler = joblib.load('feature_scaler.pkl')
    print(f"  ✅ Scaler loaded: {type(scaler).__name__}")
    
    print("\n✓ Loading encoders...")
    encoders = joblib.load('label_encoders.pkl')
    print(f"  ✅ Encoders loaded: {len(encoders)} categorical features")
    
    print("\n✓ Loading feature names...")
    features = joblib.load('feature_names.pkl')
    print(f"  ✅ Features loaded: {len(features)} total features")
    
    print("\n✓ Loading model summary...")
    summary = joblib.load('model_summary_optimized.pkl')
    print(f"  ✅ Summary loaded")
    print(f"     Model Type: {summary['model_type']}")
    print(f"     Test R²: {summary['optimized_test_r2_score']:.4f}")
    print(f"     Test RMSE: {summary['optimized_test_rmse']:.4f}")
    
    print("\n" + "=" * 80)
    print("✨ ALL SYSTEMS GO! READY FOR PRODUCTION!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)
