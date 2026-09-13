import unittest
import json
from app import app, preprocessor, model_manager, decision_engine

class TestInsightIQSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_preprocessor(self):
        cohort_dfs, combined_df = preprocessor.load_raw_datasets()
        self.assertGreater(len(cohort_dfs), 0, "Should load at least one cohort")
        self.assertGreaterEqual(len(combined_df), 900, "Combined dataset should have ~920 rows")
        
        sample_input = {
            'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 240,
            'fbs': 0, 'restecg': 0, 'thalach': 155, 'exang': 0,
            'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 3
        }
        res = preprocessor.transform_single(sample_input)
        self.assertEqual(len(res['features_imputed']), 13)
        self.assertEqual(len(res['features_scaled']), 13)
        self.assertIn('is_anomaly', res)

    def test_02_model_prediction(self):
        sample_input = {
            'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 290,
            'fbs': 1, 'restecg': 2, 'thalach': 110, 'exang': 1,
            'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 7
        }
        pred = model_manager.predict_single(sample_input)
        self.assertGreaterEqual(pred['risk_score'], 0.0)
        self.assertLessEqual(pred['risk_score'], 100.0)
        self.assertIn('random_forest', pred['model_breakdown'])
        self.assertIn('gradient_boosting', pred['model_breakdown'])
        self.assertIn('logistic_regression', pred['model_breakdown'])
        self.assertIn('extra_trees', pred['model_breakdown'])
        self.assertGreater(len(pred['feature_drivers']), 0)

    def test_03_decision_engine(self):
        sample_input = {
            'age': 35, 'sex': 0, 'cp': 1, 'trestbps': 110, 'chol': 170,
            'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0,
            'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
        }
        dec = decision_engine.evaluate_decision(sample_input)
        self.assertEqual(dec['risk_tier'], 'Low Risk')
        self.assertIn('clinical_actions', dec['recommendations'])
        self.assertIn('pharmacotherapy', dec['recommendations'])
        self.assertIn('lifestyle_modifications', dec['recommendations'])

    def test_04_whatif_simulator(self):
        base_input = {
            'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 170, 'chol': 310,
            'fbs': 1, 'restecg': 2, 'thalach': 115, 'exang': 1,
            'oldpeak': 3.0, 'slope': 2, 'ca': 2, 'thal': 7
        }
        modifications = {
            'trestbps': 120,
            'chol': 180,
            'exang': 0,
            'oldpeak': 0.2
        }
        res = decision_engine.simulate_whatif(base_input, modifications)
        self.assertTrue(res['is_improved'], "Simulated risk should improve after interventions")
        self.assertGreater(res['absolute_reduction'], 0)

    def test_05_api_endpoints(self):
        # GET /
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

        # GET /api/overview
        resp = self.client.get('/api/overview')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('kpis', data)
        self.assertGreater(data['kpis']['total_ingested_records'], 900)

        # POST /api/predict
        resp = self.client.post('/api/predict', json={
            'age': 52, 'sex': 1, 'cp': 3, 'trestbps': 135, 'chol': 230,
            'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 0.6, 'slope': 1, 'ca': 0, 'thal': 3
        })
        self.assertEqual(resp.status_code, 200)
        pred_data = resp.get_json()
        self.assertTrue(pred_data['success'])
        self.assertIn('risk_score', pred_data['data'])

        # POST /api/simulate-whatif
        resp = self.client.post('/api/simulate-whatif', json={
            'base_inputs': {'age': 60, 'trestbps': 160, 'chol': 280, 'exang': 1, 'oldpeak': 2.0},
            'modifications': {'trestbps': 120, 'chol': 190, 'exang': 0, 'oldpeak': 0.2}
        })
        self.assertEqual(resp.status_code, 200)
        sim_data = resp.get_json()
        self.assertTrue(sim_data['success'])

        # GET /api/models
        resp = self.client.get('/api/models')
        self.assertEqual(resp.status_code, 200)
        models_data = resp.get_json()
        self.assertIn('random_forest', models_data['models'])

        # GET /api/batch-data
        resp = self.client.get('/api/batch-data?limit=20')
        self.assertEqual(resp.status_code, 200)
        batch_data = resp.get_json()
        self.assertEqual(len(batch_data['records']), 20)

if __name__ == '__main__':
    unittest.main()
