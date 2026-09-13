import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

ALL_COLUMNS = FEATURE_COLUMNS + ['num']

FEATURE_LABELS = {
    'age': 'Age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest Pain Type (1-4)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1/0)',
    'restecg': 'Resting ECG Results (0-2)',
    'thalach': 'Max Heart Rate Achieved (bpm)',
    'exang': 'Exercise Induced Angina (1/0)',
    'oldpeak': 'ST Depression (Exercise vs Rest)',
    'slope': 'Slope of Peak ST Segment (1-3)',
    'ca': 'Major Vessels Colored by Fluoroscopy (0-3)',
    'thal': 'Thallium Stress Test (3, 6, 7)'
}

FEATURE_DEFAULTS = {
    'age': 54.0,
    'sex': 1.0,
    'cp': 3.0,
    'trestbps': 130.0,
    'chol': 240.0,
    'fbs': 0.0,
    'restecg': 0.0,
    'thalach': 150.0,
    'exang': 0.0,
    'oldpeak': 0.8,
    'slope': 1.0,
    'ca': 0.0,
    'thal': 3.0
}

class DataPreprocessor:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.feature_names = FEATURE_COLUMNS
        self.dataset_meta = {}

    def load_from_cache_or_prepare(self):
        """Loads fitted imputer and scaler from cache or prepares raw data."""
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'preprocessor.joblib')
        if os.path.exists(cache_path):
            try:
                import joblib
                cached = joblib.load(cache_path)
                self.imputer = cached.imputer
                self.scaler = cached.scaler
                self.anomaly_detector = cached.anomaly_detector
                self.dataset_meta = cached.dataset_meta
                return
            except Exception:
                pass
        self.prepare_data('cleveland')

    def load_raw_datasets(self):
        """Loads Cleveland (primary benchmark) and combined multi-center datasets."""
        files = {
            'cleveland': os.path.join(self.data_dir, 'processed.cleveland.data'),
            'hungarian': os.path.join(self.data_dir, 'processed.hungarian.data'),
            'switzerland': os.path.join(self.data_dir, 'processed.switzerland.data'),
            'va': os.path.join(self.data_dir, 'processed.va.data')
        }
        
        cohort_dfs = {}
        for name, path in files.items():
            if os.path.exists(path):
                df = pd.read_csv(path, header=None, names=ALL_COLUMNS, na_values=['?', '-9.0', -9.0])
                df['cohort'] = name.capitalize()
                cohort_dfs[name] = df
        
        combined_df = pd.concat(list(cohort_dfs.values()), ignore_index=True)
        return cohort_dfs, combined_df

    def prepare_data(self, dataset_key='cleveland'):
        """Prepares feature matrix X and target y for the selected dataset."""
        cohort_dfs, combined_df = self.load_raw_datasets()
        
        if dataset_key == 'combined':
            df = combined_df.copy()
        elif dataset_key in cohort_dfs:
            df = cohort_dfs[dataset_key].copy()
        else:
            df = cohort_dfs.get('cleveland', combined_df).copy()

        # Binary classification target (0 = No Heart Disease, 1 = Presence of Heart Disease)
        y = (df['num'] > 0).astype(int).values
        # Multiclass severity (0, 1, 2, 3, 4)
        y_severity = df['num'].fillna(0).astype(int).values

        X_raw = df[FEATURE_COLUMNS]
        
        # Fit imputer and transform
        X_imputed = self.imputer.fit_transform(X_raw)
        
        # Fit anomaly detector
        self.anomaly_detector.fit(X_imputed)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.dataset_meta = {
            'total_samples': len(df),
            'features': FEATURE_COLUMNS,
            'positive_count': int(np.sum(y)),
            'negative_count': int(len(y) - np.sum(y)),
            'prevalence_pct': round(float(np.mean(y) * 100), 1),
            'dataset_key': dataset_key,
            'feature_medians': {col: float(self.imputer.statistics_[i]) for i, col in enumerate(FEATURE_COLUMNS)}
        }

        return {
            'X_raw': X_raw,
            'X_imputed': X_imputed,
            'X_scaled': X_scaled,
            'y': y,
            'y_severity': y_severity,
            'df': df
        }

    def transform_single(self, input_dict):
        """Transforms a single patient input dictionary into scaled features."""
        row = []
        for col in FEATURE_COLUMNS:
            val = input_dict.get(col, FEATURE_DEFAULTS[col])
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(FEATURE_DEFAULTS[col])
        
        df_single = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        # Impute if any missing
        arr_imp = self.imputer.transform(df_single)
        # Scaled
        arr_scaled = self.scaler.transform(arr_imp)
        # Anomaly score: 1 = normal, -1 = anomaly
        is_anomaly = bool(self.anomaly_detector.predict(arr_imp)[0] == -1)
        anomaly_score = float(self.anomaly_detector.decision_function(arr_imp)[0])

        return {
            'features_imputed': arr_imp[0],
            'features_scaled': arr_scaled[0],
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 4)
        }

    def detect_batch_anomalies(self, X_imputed):
        """Returns boolean array of anomalies for a batch of imputed records."""
        preds = self.anomaly_detector.predict(X_imputed)
        scores = self.anomaly_detector.decision_function(X_imputed)
        return preds == -1, scores
