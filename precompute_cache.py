import os
import json
import joblib
from engine.preprocessor import DataPreprocessor
from engine.models import ModelManager

print("[InsightIQ Precompute] Initializing and training models...")
preprocessor = DataPreprocessor()
model_manager = ModelManager(preprocessor)
eval_results = model_manager.train_and_evaluate(dataset_key='cleveland')

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engine', 'cache')
os.makedirs(cache_dir, exist_ok=True)

models_path = os.path.join(cache_dir, 'models.joblib')
preprocessor_path = os.path.join(cache_dir, 'preprocessor.joblib')
eval_path = os.path.join(cache_dir, 'eval_results.json')
feature_imp_path = os.path.join(cache_dir, 'feature_importance.json')

joblib.dump(model_manager.fitted_models, models_path)
joblib.dump(preprocessor, preprocessor_path)

with open(eval_path, 'w') as f:
    json.dump(eval_results, f, indent=2)

with open(feature_imp_path, 'w') as f:
    json.dump(model_manager.global_feature_importance, f, indent=2)

print(f"[InsightIQ Precompute] Successfully cached models and preprocessors to {cache_dir}!")
