import numpy as np

class DecisionEngine:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def evaluate_decision(self, input_dict):
        """Processes patient inputs through ML models and generates prioritized decision support."""
        pred = self.model_manager.predict_single(input_dict)
        risk_score = pred['risk_score']
        
        # Risk Stratification
        if risk_score < 25.0:
            risk_tier = 'Low Risk'
            risk_color = '#10B981' # Emerald Safe
            triage_level = 'Level 4: Routine Wellness / Primary Prevention'
            urgency = 'Standard Vigilance'
            summary = 'Low cardiovascular risk profile. Findings are within standard limits; primary prevention and regular screening advised.'
        elif risk_score < 50.0:
            risk_tier = 'Moderate Risk'
            risk_color = '#F59E0B' # Amber Caution
            triage_level = 'Level 3: Outpatient Cardiology Evaluation'
            urgency = 'Elevated Vigilance'
            summary = 'Moderate cardiovascular risk markers detected. Actionable lifestyle intervention and subclinical atherosclerosis screening recommended.'
        elif risk_score < 75.0:
            risk_tier = 'High Risk'
            risk_color = '#F97316' # Orange Warning
            triage_level = 'Level 2: Urgent Diagnostic Referral'
            urgency = 'High Priority'
            summary = 'High probability of significant coronary artery narrowing (>50%). Comprehensive diagnostic workup and medical therapy initiation indicated.'
        else:
            risk_tier = 'Critical Risk'
            risk_color = '#EF4444' # Crimson Urgent
            triage_level = 'Level 1: Immediate Inpatient Triage / High Acuity'
            urgency = 'Critical Immediate'
            summary = 'Critical coronary ischemia indicators present with high model consensus. Requires prompt clinical assessment to prevent acute coronary syndrome.'

        # Extract specific biomarker values
        age = float(input_dict.get('age', 54))
        trestbps = float(input_dict.get('trestbps', 130))
        chol = float(input_dict.get('chol', 240))
        thalach = float(input_dict.get('thalach', 150))
        exang = float(input_dict.get('exang', 0))
        oldpeak = float(input_dict.get('oldpeak', 0.8))
        ca = float(input_dict.get('ca', 0))
        thal = float(input_dict.get('thal', 3))
        cp = float(input_dict.get('cp', 3))

        # Generate Actionable Prescriptive Recommendations
        clinical_actions = []
        pharmacotherapy = []
        lifestyle_modifications = []

        # 1. Clinical Diagnostic Actions
        if risk_score >= 60.0 or ca >= 2 or oldpeak >= 2.0:
            clinical_actions.append({
                'action': 'Schedule Invasive Coronary Angiography (ICA)',
                'priority': 'High Priority',
                'rationale': f'Pronounced ischemic markers (ST depression {oldpeak} mm, fluoroscopy score {int(ca)}) suggest multi-vessel disease.'
            })
            clinical_actions.append({
                'action': 'Serial High-Sensitivity Cardiac Troponin (hs-cTn)',
                'priority': 'Urgent',
                'rationale': 'Rule out active myocardial injury or acute coronary syndrome progression.'
            })
        elif risk_score >= 35.0:
            clinical_actions.append({
                'action': 'Order Exercise Myocardial Perfusion SPECT or Stress Echo',
                'priority': 'Recommended',
                'rationale': 'Assess functional ischemic burden under physical stress non-invasively.'
            })
            clinical_actions.append({
                'action': 'Coronary Artery Calcium (CAC) CT Scoring',
                'priority': 'Recommended',
                'rationale': 'Quantify anatomical atherosclerotic plaque burden for definitive risk recalibration.'
            })
        else:
            clinical_actions.append({
                'action': 'Annual Preventive Cardiovascular Risk Check',
                'priority': 'Routine',
                'rationale': 'Patient currently within healthy cardiovascular margins; continue standard biennial lipid panel and BP assessment.'
            })

        # 2. Pharmacotherapy & Clinical Monitoring
        if trestbps >= 140.0:
            pharmacotherapy.append({
                'title': 'Anti-Hypertensive Regimen Optimization',
                'detail': f'Resting systolic BP is {int(trestbps)} mmHg (Stage 2 Hypertension). Target BP < 120/80 mmHg using ACEi/ARB or CCB therapy.'
            })
        elif trestbps >= 130.0:
            pharmacotherapy.append({
                'title': 'Blood Pressure Monitoring Protocol',
                'detail': f'Resting systolic BP {int(trestbps)} mmHg indicates Stage 1 Hypertension. Initiate 24-hour ambulatory blood pressure monitoring (ABPM).'
            })

        if chol >= 240.0:
            pharmacotherapy.append({
                'title': 'High-Intensity Statin Consideration',
                'detail': f'Total cholesterol {int(chol)} mg/dL is elevated. Evaluate LDL-C and discuss Atorvastatin 20-40mg or Rosuvastatin 10-20mg therapy.'
            })
        elif chol >= 200.0:
            pharmacotherapy.append({
                'title': 'Lipid Panel Reassessment & Dietary Trial',
                'detail': f'Borderline elevated cholesterol ({int(chol)} mg/dL). 3-month lifestyle modification trial followed by ApoB/LDL fractionation.'
            })

        if exang == 1.0 or cp == 4:
            pharmacotherapy.append({
                'title': 'Anti-Anginal Symptom Management',
                'detail': 'Exercise-induced angina noted. Discuss Beta-blockers (e.g. Metoprolol) and sublingual nitroglycerin PRN.'
            })

        # 3. Targeted Lifestyle & Behavioral Prescriptions
        if trestbps >= 130.0:
            lifestyle_modifications.append({
                'category': 'Dietary Sodium & Potassium',
                'prescription': 'DASH Diet: Restrict sodium to < 1,500 mg/day and increase dietary potassium via leafy greens and fruits.'
            })
        if chol >= 200.0:
            lifestyle_modifications.append({
                'category': 'Saturated Fat & Fiber',
                'prescription': 'Limit dietary saturated fats to < 6% of total caloric intake; add 10-25g soluble fiber daily (psyllium, legumes, oats).'
            })
        
        if exang == 1.0 or risk_score >= 60.0:
            lifestyle_modifications.append({
                'category': 'Supervised Cardiac Rehab',
                'prescription': 'Avoid sudden unconditioned strenuous exertion. Enroll in medically monitored Phase II Cardiac Rehabilitation.'
            })
        else:
            lifestyle_modifications.append({
                'category': 'Aerobic Conditioning',
                'prescription': f'Target 150-300 min/week moderate-intensity aerobic exercise. Maintain target training heart rate around {int(thalach * 0.75)} bpm.'
            })

        return {
            'risk_score': risk_score,
            'risk_tier': risk_tier,
            'risk_color': risk_color,
            'triage_level': triage_level,
            'urgency': urgency,
            'summary': summary,
            'agreement_pct': pred['agreement_pct'],
            'is_anomaly': pred['is_anomaly'],
            'anomaly_score': pred['anomaly_score'],
            'model_breakdown': pred['model_breakdown'],
            'feature_drivers': pred['feature_drivers'],
            'recommendations': {
                'clinical_actions': clinical_actions,
                'pharmacotherapy': pharmacotherapy,
                'lifestyle_modifications': lifestyle_modifications
            }
        }

    def simulate_whatif(self, base_inputs, modifications):
        """Simulates how targeted interventions reduce the patient's cardiovascular risk score."""
        # Baseline
        base_result = self.evaluate_decision(base_inputs)
        base_risk = base_result['risk_score']

        # Modified inputs
        mod_inputs = dict(base_inputs)
        mod_inputs.update(modifications)
        
        sim_result = self.evaluate_decision(mod_inputs)
        sim_risk = sim_result['risk_score']

        absolute_reduction = round(base_risk - sim_risk, 1)
        relative_reduction = round((absolute_reduction / base_risk) * 100, 1) if base_risk > 0 else 0.0

        return {
            'base_risk': base_risk,
            'sim_risk': sim_risk,
            'absolute_reduction': absolute_reduction,
            'relative_reduction': relative_reduction,
            'base_tier': base_result['risk_tier'],
            'sim_tier': sim_result['risk_tier'],
            'is_improved': sim_risk < base_risk,
            'modifications_applied': modifications,
            'takeaway': (
                f"Modifying these parameters reduced cardiovascular risk by {absolute_reduction}% "
                f"({relative_reduction}% relative reduction), successfully transitioning the patient "
                f"from {base_result['risk_tier']} to {sim_result['risk_tier']}."
            )
        }

    def get_sample_personas(self):
        """Returns verified real-world clinical archetypes for instant evaluation."""
        return [
            {
                'id': 'athlete_low_risk',
                'name': 'Marcus Vance - Active Runner',
                'tag': 'Low Risk Baseline',
                'description': '38-year-old marathon runner with optimal blood pressure and high cardiovascular reserve.',
                'attributes': {
                    'age': 38, 'sex': 1, 'cp': 2, 'trestbps': 114, 'chol': 175,
                    'fbs': 0, 'restecg': 0, 'thalach': 188, 'exang': 0,
                    'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
                }
            },
            {
                'id': 'executive_moderate_risk',
                'name': 'Eleanor Brooks - Corporate Director',
                'tag': 'Moderate Risk / Borderline',
                'description': '52-year-old executive experiencing stress, Stage 1 hypertension, and borderline hyperlipidemia.',
                'attributes': {
                    'age': 52, 'sex': 0, 'cp': 3, 'trestbps': 136, 'chol': 238,
                    'fbs': 0, 'restecg': 1, 'thalach': 152, 'exang': 0,
                    'oldpeak': 0.8, 'slope': 2, 'ca': 0, 'thal': 3
                }
            },
            {
                'id': 'smoker_high_risk',
                'name': 'Arthur Pendelton - Hypertensive Smoker',
                'tag': 'High Risk Alert',
                'description': '61-year-old with longstanding hypertension, elevated cholesterol, and exertional angina.',
                'attributes': {
                    'age': 61, 'sex': 1, 'cp': 4, 'trestbps': 158, 'chol': 282,
                    'fbs': 1, 'restecg': 2, 'thalach': 125, 'exang': 1,
                    'oldpeak': 2.2, 'slope': 2, 'ca': 1, 'thal': 7
                }
            },
            {
                'id': 'emergency_critical_risk',
                'name': 'Geraldine O’Connor - Acute Ischemia',
                'tag': 'Critical Urgency Triage',
                'description': '67-year-old patient admitted with asymptomatic ischemic episodes, ST depression > 3.0mm, and fluoroscopy showing 3 colored vessels.',
                'attributes': {
                    'age': 67, 'sex': 0, 'cp': 4, 'trestbps': 172, 'chol': 315,
                    'fbs': 1, 'restecg': 2, 'thalach': 108, 'exang': 1,
                    'oldpeak': 3.4, 'slope': 3, 'ca': 3, 'thal': 7
                }
            }
        ]
