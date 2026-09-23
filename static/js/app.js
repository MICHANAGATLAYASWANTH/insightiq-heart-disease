/**
 * InsightIQ - Frontend Application Logic
 * "Turning data into intelligent decisions"
 */

document.addEventListener('DOMContentLoaded', () => {
    // Dynamic API Base URL for GitHub Pages / Vercel cross-compatibility
    const API_BASE = (window.location.hostname.includes('github.io') || window.location.protocol === 'file:')
        ? 'https://insightiq-heart-disease.vercel.app'
        : '';

    // Application State
    const state = {
        currentPatientInputs: {},
        samplePersonas: [],
        charts: {},
        batchRecords: [],
        activeTab: 'simulator-tab',
        whatIfDebounceTimer: null
    };

    // DOM Elements
    const elements = {
        themeToggle: document.getElementById('theme-toggle'),
        tabButtons: document.querySelectorAll('.tab-btn'),
        tabPanes: document.querySelectorAll('.tab-pane'),
        predictionForm: document.getElementById('prediction-form'),
        personaContainer: document.getElementById('persona-buttons-container'),
        btnResetForm: document.getElementById('btn-reset-form'),
        
        // Output Elements
        riskScoreText: document.getElementById('res-risk-score'),
        riskTierBadge: document.getElementById('res-risk-tier'),
        triageLevelText: document.getElementById('res-triage-level'),
        triageSummary: document.getElementById('res-triage-summary'),
        agreementPct: document.getElementById('res-agreement-pct'),
        gaugeMeter: document.getElementById('gauge-meter-fill'),
        anomalyBadge: document.getElementById('anomaly-badge'),
        
        // Model Pill Values
        valRF: document.getElementById('m-val-rf'),
        valGB: document.getElementById('m-val-gb'),
        valLR: document.getElementById('m-val-lr'),
        valET: document.getElementById('m-val-et'),

        // Lists
        driversList: document.getElementById('drivers-list-container'),
        listClinicalActions: document.getElementById('list-clinical-actions'),
        listPharmacotherapy: document.getElementById('list-pharmacotherapy'),
        listLifestyle: document.getElementById('list-lifestyle'),

        // What-If Elements
        sliderBP: document.getElementById('whatif-bp'),
        sliderChol: document.getElementById('whatif-chol'),
        sliderExang: document.getElementById('whatif-exang'),
        sliderOldpeak: document.getElementById('whatif-oldpeak'),
        valWhatIfBP: document.getElementById('val-whatif-bp'),
        valWhatIfChol: document.getElementById('val-whatif-chol'),
        valWhatIfExang: document.getElementById('val-whatif-exang'),
        valWhatIfOldpeak: document.getElementById('val-whatif-oldpeak'),
        whatIfDeltaTag: document.getElementById('whatif-delta-tag'),
        whatIfDeltaVal: document.getElementById('whatif-delta-val'),
        whatIfTakeaway: document.getElementById('whatif-takeaway'),

        // Batch Elements
        batchSearchInput: document.getElementById('batch-search-input'),
        filterRiskTier: document.getElementById('filter-risk-tier'),
        filterCohort: document.getElementById('filter-cohort'),
        batchTableBody: document.getElementById('batch-table-body'),
        fileInput: document.getElementById('batch-file-input'),
        btnExportBatch: document.getElementById('btn-export-batch'),

        // Modal
        modal: document.getElementById('patient-modal'),
        modalContent: document.getElementById('modal-patient-content'),
        modalPatientId: document.getElementById('modal-patient-id'),
        btnCloseModal: document.getElementById('btn-close-modal')
    };

    // =========================================================================
    // Initialization
    // =========================================================================
    initTheme();
    initTabs();
    loadOverviewKPIs();
    loadSamplePersonas();
    loadModelObservatory();
    loadBatchData();
    setupWhatIfListeners();

    // Trigger initial prediction on default values
    submitPredictionForm();

    // =========================================================================
    // Theme Switcher
    // =========================================================================
    function initTheme() {
        const savedTheme = localStorage.getItem('insightiq-theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            elements.themeToggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
        }

        elements.themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            elements.themeToggle.innerHTML = isLight 
                ? '<i class="fa-solid fa-sun"></i>' 
                : '<i class="fa-solid fa-moon"></i>';
            localStorage.setItem('insightiq-theme', isLight ? 'light' : 'dark');
            
            // Re-render chart colors if necessary
            Object.values(state.charts).forEach(c => c && c.update && c.update());
        });
    }

    // =========================================================================
    // Navigation & Tabs
    // =========================================================================
    function initTabs() {
        elements.tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTabId = btn.getAttribute('data-tab');
                elements.tabButtons.forEach(b => b.classList.remove('active'));
                elements.tabPanes.forEach(p => p.classList.remove('active'));

                btn.classList.add('active');
                const pane = document.getElementById(targetTabId);
                if (pane) pane.classList.add('active');
                state.activeTab = targetTabId;

                // Refresh charts if entering tab
                if (targetTabId === 'overview-tab' || targetTabId === 'models-tab') {
                    setTimeout(() => {
                        Object.values(state.charts).forEach(c => c && c.resize && c.resize());
                    }, 50);
                }
            });
        });
    }



    // =========================================================================
    // Sample Personas
    // =========================================================================
    async function loadSamplePersonas() {
        try {
            const res = await fetch(`${API_BASE}/api/sample-personas`);
            const data = await res.json();
            state.samplePersonas = data.personas || [];
            renderPersonaChips();
        } catch (err) {
            console.error('Failed to load personas:', err);
        }
    }

    function renderPersonaChips() {
        if (!elements.personaContainer) return;
        elements.personaContainer.innerHTML = '';
        state.samplePersonas.forEach(p => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'persona-btn';
            btn.innerHTML = `<i class="fa-solid fa-user-check"></i> ${p.name.split(' - ')[0]}`;
            btn.title = `${p.tag}: ${p.description}`;
            btn.addEventListener('click', () => fillPersona(p));
            elements.personaContainer.appendChild(btn);
        });
    }

    function fillPersona(persona) {
        const attrs = persona.attributes;
        for (const [key, val] of Object.entries(attrs)) {
            const input = document.getElementById(`input-${key}`);
            if (input) input.value = val;
        }
        submitPredictionForm();
    }

    elements.btnResetForm.addEventListener('click', () => {
        elements.predictionForm.reset();
        submitPredictionForm();
    });

    // =========================================================================
    // Prediction Form Submission
    // =========================================================================
    elements.predictionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitPredictionForm();
    });

    function getFormValues() {
        const formData = new FormData(elements.predictionForm);
        const inputs = {};
        for (const [key, val] of formData.entries()) {
            inputs[key] = parseFloat(val);
        }
        return inputs;
    }

    async function submitPredictionForm() {
        const inputs = getFormValues();
        state.currentPatientInputs = inputs;

        // Visual loading state
        elements.riskScoreText.textContent = '...';
        elements.riskTierBadge.textContent = 'Evaluating ML Models...';

        try {
            const res = await fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputs)
            });
            const json = await res.json();
            if (json.success && json.data) {
                renderPredictionResults(json.data);
                syncWhatIfSliders(inputs);
                runWhatIfSimulation();
            }
        } catch (err) {
            console.error('Prediction API Error:', err);
        }
    }

    function renderPredictionResults(data) {
        const score = data.risk_score;
        elements.riskScoreText.textContent = score.toFixed(1);
        elements.riskTierBadge.textContent = data.risk_tier;
        elements.riskTierBadge.style.color = data.risk_color;
        elements.riskTierBadge.style.backgroundColor = `${data.risk_color}22`;
        elements.riskTierBadge.style.borderColor = data.risk_color;

        elements.triageLevelText.textContent = data.triage_level;
        elements.triageSummary.textContent = data.summary;
        elements.agreementPct.textContent = `${data.agreement_pct}%`;

        // Update Circular Gauge (circumference = 2 * PI * 66 ~= 414.69)
        const totalCircumference = 414.69;
        const offset = totalCircumference - (score / 100) * totalCircumference;
        elements.gaugeMeter.style.strokeDashoffset = offset;
        elements.gaugeMeter.style.stroke = data.risk_color;

        // Anomaly badge
        if (data.is_anomaly) {
            elements.anomalyBadge.style.display = 'inline-flex';
            elements.anomalyBadge.title = `Isolation Forest Anomaly Score: ${data.anomaly_score}`;
        } else {
            elements.anomalyBadge.style.display = 'none';
        }

        // Model Breakdown Pills
        const mb = data.model_breakdown || {};
        if (mb.random_forest) elements.valRF.textContent = `${mb.random_forest.probability}%`;
        if (mb.gradient_boosting) elements.valGB.textContent = `${mb.gradient_boosting.probability}%`;
        if (mb.logistic_regression) elements.valLR.textContent = `${mb.logistic_regression.probability}%`;
        if (mb.extra_trees) elements.valET.textContent = `${mb.extra_trees.probability}%`;

        // Explainability Drivers
        renderDrivers(data.feature_drivers || []);

        // Actionable Prescriptions
        renderRecommendations(data.recommendations || {});
    }

    function renderDrivers(drivers) {
        elements.driversList.innerHTML = '';
        if (drivers.length === 0) {
            elements.driversList.innerHTML = '<div class="hint-text">No significant biomarker deviation from population baseline.</div>';
            return;
        }

        drivers.slice(0, 6).forEach(d => {
            const row = document.createElement('div');
            row.className = 'driver-row';
            const isRisk = d.effect === 'increases_risk';
            const sign = d.impact > 0 ? `+${d.impact}%` : `${d.impact}%`;
            
            row.innerHTML = `
                <div class="driver-info">
                    <span class="driver-label">${d.label}</span>
                    <span class="driver-sub">Patient: <strong>${d.value}</strong> &bull; Baseline Median: ${d.median}</span>
                </div>
                <span class="driver-impact-badge ${isRisk ? 'risk' : 'protective'}">
                    ${sign} ${isRisk ? 'Risk' : 'Protect'}
                </span>
            `;
            elements.driversList.appendChild(row);
        });
    }

    function renderRecommendations(recs) {
        // Clinical Actions
        elements.listClinicalActions.innerHTML = '';
        (recs.clinical_actions || []).forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.action} (${item.priority})</strong><p>${item.rationale}</p>`;
            elements.listClinicalActions.appendChild(li);
        });

        // Pharmacotherapy
        elements.listPharmacotherapy.innerHTML = '';
        (recs.pharmacotherapy || []).forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.title}</strong><p>${item.detail}</p>`;
            elements.listPharmacotherapy.appendChild(li);
        });

        // Lifestyle
        elements.listLifestyle.innerHTML = '';
        (recs.lifestyle_modifications || []).forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.category}</strong><p>${item.prescription}</p>`;
            elements.listLifestyle.appendChild(li);
        });
    }

    // =========================================================================
    // What-If Counterfactual Simulator
    // =========================================================================
    function syncWhatIfSliders(inputs) {
        if (inputs.trestbps) {
            elements.sliderBP.value = Math.min(190, Math.max(100, inputs.trestbps));
            elements.valWhatIfBP.textContent = `${elements.sliderBP.value} mm Hg`;
        }
        if (inputs.chol) {
            elements.sliderChol.value = Math.min(320, Math.max(130, inputs.chol));
            elements.valWhatIfChol.textContent = `${elements.sliderChol.value} mg/dL`;
        }
        if (inputs.exang !== undefined) {
            elements.sliderExang.value = inputs.exang;
            elements.valWhatIfExang.textContent = inputs.exang === 1 ? 'Angina Present' : 'No Symptoms';
        }
        if (inputs.oldpeak !== undefined) {
            elements.sliderOldpeak.value = inputs.oldpeak;
            elements.valWhatIfOldpeak.textContent = `${parseFloat(inputs.oldpeak).toFixed(1)} mm`;
        }
    }

    function setupWhatIfListeners() {
        const sliders = [
            { el: elements.sliderBP, label: elements.valWhatIfBP, unit: ' mm Hg' },
            { el: elements.sliderChol, label: elements.valWhatIfChol, unit: ' mg/dL' },
            { el: elements.sliderOldpeak, label: elements.valWhatIfOldpeak, unit: ' mm' }
        ];

        sliders.forEach(s => {
            s.el.addEventListener('input', () => {
                s.label.textContent = `${s.el.value}${s.unit}`;
                triggerDebouncedWhatIf();
            });
        });

        elements.sliderExang.addEventListener('input', () => {
            elements.valWhatIfExang.textContent = elements.sliderExang.value === '1' ? 'Angina Present' : 'No Symptoms';
            triggerDebouncedWhatIf();
        });
    }

    function triggerDebouncedWhatIf() {
        clearTimeout(state.whatIfDebounceTimer);
        state.whatIfDebounceTimer = setTimeout(runWhatIfSimulation, 220);
    }

    async function runWhatIfSimulation() {
        if (!state.currentPatientInputs.age) return;

        const modifications = {
            trestbps: parseFloat(elements.sliderBP.value),
            chol: parseFloat(elements.sliderChol.value),
            exang: parseFloat(elements.sliderExang.value),
            oldpeak: parseFloat(elements.sliderOldpeak.value)
        };

        try {
            const res = await fetch(`${API_BASE}/api/simulate-whatif`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base_inputs: state.currentPatientInputs,
                    modifications: modifications
                })
            });
            const json = await res.json();
            if (json.success && json.data) {
                const d = json.data;
                const red = d.absolute_reduction;
                elements.whatIfDeltaVal.textContent = `${red >= 0 ? '-' : '+'}${Math.abs(red).toFixed(1)}%`;
                
                if (red > 0) {
                    elements.whatIfDeltaTag.style.background = 'rgba(16, 185, 129, 0.15)';
                    elements.whatIfDeltaTag.style.color = '#10B981';
                    elements.whatIfDeltaTag.style.borderColor = 'rgba(16, 185, 129, 0.35)';
                } else {
                    elements.whatIfDeltaTag.style.background = 'rgba(239, 68, 68, 0.15)';
                    elements.whatIfDeltaTag.style.color = '#EF4444';
                    elements.whatIfDeltaTag.style.borderColor = 'rgba(239, 68, 68, 0.35)';
                }

                elements.whatIfTakeaway.innerHTML = `<i class="fa-solid fa-lightbulb"></i> ${d.takeaway}`;
            }
        } catch (err) {
            console.error('What-If simulation error:', err);
        }
    }

    // =========================================================================
    // Population Analytics & Charts
    // =========================================================================
    async function loadOverviewKPIs() {
        try {
            const res = await fetch(`${API_BASE}/api/overview`);
            const data = await res.json();
            if (!data.kpis) return;

            document.getElementById('kpi-total-records').textContent = data.kpis.total_ingested_records;
            document.getElementById('kpi-roc-auc').textContent = `${data.kpis.ensemble_roc_auc}%`;
            document.getElementById('kpi-alerts-count').textContent = data.kpis.high_risk_alerts;
            document.getElementById('kpi-anomalies-count').textContent = data.kpis.detected_anomalies;

            renderRiskDistributionChart(data.risk_distribution);
            renderCohortsChart(data.cohort_distribution);
        } catch (err) {
            console.error('Failed to load overview KPIs:', err);
        }
    }

    function renderRiskDistributionChart(dist) {
        const ctx = document.getElementById('chart-risk-distribution');
        if (!ctx) return;

        if (state.charts.riskDist) state.charts.riskDist.destroy();

        state.charts.riskDist = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(dist),
                datasets: [{
                    data: Object.values(dist),
                    backgroundColor: ['#10B981', '#F59E0B', '#F97316', '#EF4444'],
                    borderColor: 'rgba(15, 23, 42, 0.8)',
                    borderWidth: 3,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', weight: '600' } }
                    }
                },
                cutout: '68%'
            }
        });
    }

    function renderCohortsChart(cohorts) {
        const ctx = document.getElementById('chart-cohorts');
        if (!ctx) return;

        if (state.charts.cohorts) state.charts.cohorts.destroy();

        state.charts.cohorts = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(cohorts),
                datasets: [{
                    label: 'Patient Count',
                    data: Object.values(cohorts),
                    backgroundColor: 'rgba(99, 102, 241, 0.75)',
                    borderColor: '#6366F1',
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
                }
            }
        });
    }

    // =========================================================================
    // Model Observatory & Benchmark Matrix
    // =========================================================================
    async function loadModelObservatory() {
        try {
            const res = await fetch(`${API_BASE}/api/models`);
            const data = await res.json();
            if (!data.models) return;

            renderBenchmarkTable(data.models);
            renderROCCurvesChart(data.models);
            renderConfusionMatrix(data.models.random_forest ? data.models.random_forest.confusion_matrix : null);
            renderGlobalImportanceChart(data.global_feature_importance || []);
        } catch (err) {
            console.error('Failed to load models data:', err);
        }
    }

    function renderBenchmarkTable(models) {
        const tbody = document.getElementById('benchmark-table-body');
        if (!tbody) return;
        tbody.innerHTML = '';

        Object.values(models).forEach(m => {
            const cm = m.confusion_matrix || {};
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${m.name}</strong></td>
                <td><span class="metric-highlight">${m.accuracy}%</span></td>
                <td>${m.precision}%</td>
                <td>${m.recall}%</td>
                <td>${m.f1}%</td>
                <td><span class="metric-highlight">${m.roc_auc}%</span></td>
                <td>${cm.tn || 0}</td>
                <td>${cm.fp || 0}</td>
                <td>${cm.fn || 0}</td>
                <td>${cm.tp || 0}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderROCCurvesChart(models) {
        const ctx = document.getElementById('chart-roc-curves');
        if (!ctx) return;

        if (state.charts.roc) state.charts.roc.destroy();

        const colors = {
            random_forest: '#6366F1',
            gradient_boosting: '#06B6D4',
            logistic_regression: '#F59E0B',
            extra_trees: '#10B981'
        };

        const datasets = Object.keys(models).map(key => {
            const m = models[key];
            const points = (m.roc_curve || []).map(pt => ({ x: pt.fpr, y: pt.tpr }));
            return {
                label: `${m.name} (AUC ${m.roc_auc}%)`,
                data: points,
                borderColor: colors[key] || '#6366F1',
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 0
            };
        });

        // Baseline diagonal
        datasets.push({
            label: 'Random Guess Baseline (AUC 50%)',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0
        });

        state.charts.roc = new Chart(ctx, {
            type: 'line',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', boxWidth: 12, font: { size: 11 } }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' },
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.06)' }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'True Positive Rate (Sensitivity / Recall)', color: '#94a3b8' },
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.06)' }
                    }
                }
            }
        });
    }

    function renderConfusionMatrix(cm) {
        const wrapper = document.getElementById('cm-display-wrapper');
        if (!wrapper || !cm) return;

        const total = cm.tn + cm.fp + cm.fn + cm.tp;
        const sensitivity = ((cm.tp / (cm.tp + cm.fn)) * 100).toFixed(1);
        const specificity = ((cm.tn / (cm.tn + cm.fp)) * 100).toFixed(1);

        wrapper.innerHTML = `
            <div class="cm-grid">
                <div class="cm-cell negative-match">
                    <span class="cm-count">${cm.tn}</span>
                    <span class="cm-label">True Negative (Healthy)</span>
                </div>
                <div class="cm-cell error-match">
                    <span class="cm-count">${cm.fp}</span>
                    <span class="cm-label">False Positive (False Alarm)</span>
                </div>
                <div class="cm-cell error-match">
                    <span class="cm-count">${cm.fn}</span>
                    <span class="cm-label">False Negative (Missed Disease)</span>
                </div>
                <div class="cm-cell positive-match">
                    <span class="cm-count">${cm.tp}</span>
                    <span class="cm-label">True Positive (Confirmed Disease)</span>
                </div>
            </div>
            <div style="margin-top: 1rem; font-size: 0.82rem; color: #94a3b8; text-align: center;">
                Diagnostic Sensitivity: <strong style="color: #10B981;">${sensitivity}%</strong> &bull; 
                Specificity: <strong style="color: #06B6D4;">${specificity}%</strong> &bull; 
                Total Evaluated Cohort: <strong>${total}</strong>
            </div>
        `;
    }

    function renderGlobalImportanceChart(importanceList) {
        const ctx = document.getElementById('chart-global-importance');
        if (!ctx) return;

        if (state.charts.importance) state.charts.importance.destroy();

        const labels = importanceList.map(i => i.label.split('(')[0].trim());
        const values = importanceList.map(i => i.importance);

        state.charts.importance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Relative Feature Importance (%)',
                    data: values,
                    backgroundColor: 'rgba(6, 182, 212, 0.75)',
                    borderColor: '#06B6D4',
                    borderWidth: 1.5,
                    borderRadius: 5
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
                    y: { ticks: { color: '#f8fafc', font: { size: 11, weight: '600' } }, grid: { display: false } }
                }
            }
        });
    }

    // =========================================================================
    // Batch Ingestion & Anomaly Auditing
    // =========================================================================
    async function loadBatchData() {
        try {
            const res = await fetch(`${API_BASE}/api/batch-data?limit=150`);
            const data = await res.json();
            state.batchRecords = data.records || [];
            renderBatchTable(state.batchRecords);
            setupBatchFilters();
        } catch (err) {
            console.error('Failed to load batch records:', err);
        }
    }

    function renderBatchTable(records) {
        if (!elements.batchTableBody) return;
        elements.batchTableBody.innerHTML = '';

        if (records.length === 0) {
            elements.batchTableBody.innerHTML = '<tr><td colspan="12" class="text-center">No matching records found.</td></tr>';
            return;
        }

        records.forEach(r => {
            const tr = document.createElement('tr');
            const tierClass = r.risk_tier.toLowerCase().replace(' risk', '');
            
            tr.innerHTML = `
                <td><strong>${r.id}</strong></td>
                <td>${r.cohort}</td>
                <td>${r.age} / ${r.sex[0]}</td>
                <td>${r.chest_pain}</td>
                <td>${r.bp}</td>
                <td>${r.cholesterol}</td>
                <td>${r.max_hr}</td>
                <td>${r.st_depression}</td>
                <td><strong>${r.risk_score.toFixed(1)}%</strong></td>
                <td><span class="badge-tier-pill ${tierClass}">${r.risk_tier}</span></td>
                <td>${r.is_anomaly ? '<span class="badge-anomaly-yes"><i class="fa-solid fa-triangle-exclamation"></i> Outlier</span>' : '<span style="color:#64748b;">Normal</span>'}</td>
                <td><button class="btn-table-action" data-id="${r.id}"><i class="fa-solid fa-magnifying-glass"></i> Inspect</button></td>
            `;
            elements.batchTableBody.appendChild(tr);
        });

        // Add Inspect click events
        elements.batchTableBody.querySelectorAll('.btn-table-action').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.getAttribute('data-id');
                const record = state.batchRecords.find(item => item.id === id);
                if (record) openPatientModal(record);
            });
        });
    }

    function setupBatchFilters() {
        const applyFilters = () => {
            const query = elements.batchSearchInput.value.toLowerCase().trim();
            const tier = elements.filterRiskTier.value;
            const cohort = elements.filterCohort.value.toLowerCase();

            const filtered = state.batchRecords.filter(r => {
                const matchesQuery = !query || 
                    r.id.toLowerCase().includes(query) || 
                    r.cohort.toLowerCase().includes(query) ||
                    r.chest_pain.toLowerCase().includes(query);

                const matchesTier = tier === 'all' || r.risk_tier === tier;
                const matchesCohort = cohort === 'all' || r.cohort.toLowerCase().includes(cohort);

                return matchesQuery && matchesTier && matchesCohort;
            });

            renderBatchTable(filtered);
        };

        elements.batchSearchInput.addEventListener('input', applyFilters);
        elements.filterRiskTier.addEventListener('change', applyFilters);
        elements.filterCohort.addEventListener('change', applyFilters);

        // CSV Upload
        elements.fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch(`${API_BASE}/api/upload-csv`, { method: 'POST', body: formData });
                const json = await res.json();
                if (json.success) {
                    alert(`Successfully ingested and scored ${json.count} records from ${file.name}!`);
                    loadBatchData();
                } else {
                    alert(`Upload error: ${json.error}`);
                }
            } catch (err) {
                alert(`Upload failed: ${err.message}`);
            }
        });

        // Export Report
        elements.btnExportBatch.addEventListener('click', () => {
            if (state.batchRecords.length === 0) return;
            const headers = ['ID', 'Cohort', 'Age', 'Sex', 'ChestPain', 'BP', 'Cholesterol', 'MaxHR', 'STDepression', 'RiskScore', 'RiskTier', 'IsAnomaly'];
            const rows = state.batchRecords.map(r => [
                r.id, r.cohort, r.age, r.sex, r.chest_pain, r.bp, r.cholesterol, r.max_hr, r.st_depression, r.risk_score, r.risk_tier, r.is_anomaly
            ]);
            
            const csvContent = 'data:text/csv;charset=utf-8,' + 
                [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
            
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement('a');
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', 'InsightIQ_Batch_Risk_Audit.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // Modal
    function openPatientModal(record) {
        elements.modalPatientId.innerHTML = `<i class="fa-solid fa-user-tag"></i> Patient Dossier: ${record.id} (${record.cohort})`;
        elements.modalContent.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;">
                <div>
                    <h4 style="font-size: 1.2rem; margin-bottom: 0.2rem;">Risk Probability: <strong style="color: #6366F1;">${record.risk_score.toFixed(1)}%</strong></h4>
                    <span class="badge-tier-pill ${record.risk_tier.toLowerCase().replace(' risk', '')}">${record.risk_tier}</span>
                </div>
                ${record.is_anomaly ? '<span class="badge-anomaly-yes"><i class="fa-solid fa-triangle-exclamation"></i> Outlier Flagged</span>' : ''}
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; background: rgba(0,0,0,0.25); padding: 1rem; border-radius: 8px; font-size: 0.85rem; margin-bottom: 1.2rem;">
                <div><strong>Age / Sex:</strong> ${record.age} years / ${record.sex}</div>
                <div><strong>Chest Pain Type:</strong> ${record.chest_pain}</div>
                <div><strong>Resting Blood Pressure:</strong> ${record.bp} mm Hg</div>
                <div><strong>Serum Cholesterol:</strong> ${record.cholesterol} mg/dL</div>
                <div><strong>Max Exercise Heart Rate:</strong> ${record.max_hr} bpm</div>
                <div><strong>ST Segment Depression:</strong> ${record.st_depression} mm</div>
            </div>

            <div style="display: flex; gap: 0.8rem;">
                <button class="btn-primary" id="btn-load-to-simulator"><i class="fa-solid fa-arrow-right-to-bracket"></i> Load into Simulator</button>
            </div>
        `;

        document.getElementById('btn-load-to-simulator').addEventListener('click', () => {
            elements.modal.style.display = 'none';
            // Switch to simulator tab and load inputs
            const simBtn = document.getElementById('tab-btn-simulator');
            if (simBtn) simBtn.click();
            
            if (record.raw_inputs) {
                for (const [key, val] of Object.entries(record.raw_inputs)) {
                    const inp = document.getElementById(`input-${key}`);
                    if (inp) inp.value = val;
                }
                submitPredictionForm();
            }
        });

        elements.modal.style.display = 'flex';
    }

    elements.btnCloseModal.addEventListener('click', () => {
        elements.modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === elements.modal) elements.modal.style.display = 'none';
    });

});
