import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from engine.preprocessor import FEATURE_COLUMNS, FEATURE_LABELS

class ModelManager:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.06, max_depth=3, random_state=42),
            'logistic_regression': LogisticRegression(C=0.6, max_iter=1000, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42)
        }
        self.model_display_names = {
            'random_forest': 'Random Forest Classifier',
            'gradient_boosting': 'Gradient Boosting Machine',
            'logistic_regression': 'Calibrated Logistic Regression',
            'extra_trees': 'Extra Trees Classifier'
        }
        self.fitted_models = {}
        self.evaluation_results = {}
        self.global_feature_importance = {}
        self.primary_model_key = 'random_forest'

    def train_and_evaluate(self, dataset_key='cleveland'):
        """Trains all models with cross-validation and computes performance metrics."""
        data = self.preprocessor.prepare_data(dataset_key=dataset_key)
        X_scaled = data['X_scaled']
        X_imputed = data['X_imputed']
        y = data['y']

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        eval_results = {}

        for key, model in self.models.items():
            # For tree-based, unscaled imputed or scaled can be used; scaled works uniformly
            X_train = X_scaled if key == 'logistic_regression' else X_imputed
            
            # Cross-validated predictions
            y_pred_cv = cross_val_predict(model, X_train, y, cv=cv, method='predict')
            y_prob_cv = cross_val_predict(model, X_train, y, cv=cv, method='predict_proba')[:, 1]
            
            # Metrics
            acc = float(accuracy_score(y, y_pred_cv))
            prec = float(precision_score(y, y_pred_cv, zero_division=0))
            rec = float(recall_score(y, y_pred_cv, zero_division=0))
            f1 = float(f1_score(y, y_pred_cv, zero_division=0))
            auc = float(roc_auc_score(y, y_prob_cv))
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred_cv)
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]

            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y, y_prob_cv)
            # Sample 20 points for smooth charting payload
            step = max(1, len(fpr) // 25)
            roc_points = [
                {'fpr': round(float(fpr[i]), 3), 'tpr': round(float(tpr[i]), 3)}
                for i in range(0, len(fpr), step)
            ]
            if roc_points[-1]['fpr'] != 1.0:
                roc_points.append({'fpr': 1.0, 'tpr': 1.0})

            # Fit on full data for deployment
            model.fit(X_train, y)
            self.fitted_models[key] = model

            eval_results[key] = {
                'key': key,
                'name': self.model_display_names[key],
                'accuracy': round(acc * 100, 2),
                'precision': round(prec * 100, 2),
                'recall': round(rec * 100, 2),
                'f1': round(f1 * 100, 2),
                'roc_auc': round(auc * 100, 2),
                'confusion_matrix': {
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                    'matrix': [[tn, fp], [fn, tp]]
                },
                'roc_curve': roc_points
            }

        self.evaluation_results = eval_results

        # Calculate feature importance
        rf = self.fitted_models['random_forest']
        importances = rf.feature_importances_
        feature_importance_list = [
            {
                'feature': col,
                'label': FEATURE_LABELS[col],
                'importance': round(float(imp * 100), 2)
            }
            for col, imp in zip(FEATURE_COLUMNS, importances)
        ]
        feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
        self.global_feature_importance = feature_importance_list

        return eval_results

    def predict_single(self, input_dict):
        """Generates predictions across models and computes feature driver attribution."""
        transformed = self.preprocessor.transform_single(input_dict)
        x_imp = transformed['features_imputed'].reshape(1, -1)
        x_scaled = transformed['features_scaled'].reshape(1, -1)

        model_preds = {}
        probs = []

        for key, model in self.fitted_models.items():
            x_in = x_scaled if key == 'logistic_regression' else x_imp
            prob = float(model.predict_proba(x_in)[0, 1])
            model_preds[key] = {
                'name': self.model_display_names[key],
                'probability': round(prob * 100, 1),
                'prediction': int(prob >= 0.5)
            }
            probs.append(prob)

        # Ensemble average risk probability (weighted: RF 35%, GB 30%, LR 20%, ET 15%)
        ensemble_prob = (
            model_preds['random_forest']['probability'] * 0.35 +
            model_preds['gradient_boosting']['probability'] * 0.30 +
            model_preds['logistic_regression']['probability'] * 0.20 +
            model_preds['extra_trees']['probability'] * 0.15
        )
        ensemble_prob = round(ensemble_prob, 1)

        # Model Consensus / Agreement Score
        preds_binary = [p['prediction'] for p in model_preds.values()]
        agreement_pct = round((preds_binary.count(int(ensemble_prob >= 50.0)) / len(preds_binary)) * 100, 1)

        # Local Explainability: Feature contribution waterfall
        # We compare patient's values with population medians and model weights
        feature_drivers = self._compute_feature_drivers(transformed['features_imputed'])

        return {
            'risk_score': ensemble_prob,
            'agreement_pct': agreement_pct,
            'is_anomaly': transformed['is_anomaly'],
            'anomaly_score': transformed['anomaly_score'],
            'model_breakdown': model_preds,
            'feature_drivers': feature_drivers
        }

    def _compute_feature_drivers(self, patient_features):
        """Computes local driver attribution explaining which factors increase or decrease risk."""
        medians = [self.preprocessor.imputer.statistics_[i] for i in range(len(FEATURE_COLUMNS))]
        stds = [self.preprocessor.scaler.scale_[i] for i in range(len(FEATURE_COLUMNS))]
        
        # Logistic regression coefficients provide direct directional sign and magnitude
        lr = self.fitted_models['logistic_regression']
        coefs = lr.coef_[0]
        rf_weights = self.fitted_models['random_forest'].feature_importances_

        drivers = []
        for i, col in enumerate(FEATURE_COLUMNS):
            val = patient_features[i]
            med = medians[i]
            std = stds[i] if stds[i] != 0 else 1.0
            
            # Z-score relative to healthy population baseline
            z = (val - med) / std
            # Directional impact
            impact = z * coefs[i] * (rf_weights[i] + 0.1) * 12.0
            impact = round(float(np.clip(impact, -35.0, 35.0)), 1)

            drivers.append({
                'feature': col,
                'label': FEATURE_LABELS[col],
                'value': round(float(val), 2),
                'median': round(float(med), 2),
                'impact': impact,
                'effect': 'increases_risk' if impact > 0 else 'protective'
            })

        # Sort by absolute impact descending
        drivers.sort(key=lambda d: abs(d['impact']), reverse=True)
        return drivers
