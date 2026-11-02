(function () {
    'use strict';
    let selectedExamType = null;
    let selectedFile = null;
    let selectedSample = null;
    let examTypes = [];

    const step1 = document.getElementById('step-1');
    const step2 = document.getElementById('step-2');
    const step3 = document.getElementById('step-3');
    const examContainer = document.getElementById('exam-types-container');
    const uploadArea = document.getElementById('upload-area');
    const uploadContent = document.getElementById('upload-content');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImage = document.getElementById('preview-image');
    const fileInput = document.getElementById('file-input');
    const samplesGrid = document.getElementById('samples-grid');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultSuccess = document.getElementById('result-success');
    const resultError = document.getElementById('result-error');
    const errorMessage = document.getElementById('error-message');
    const predictionLabel = document.getElementById('prediction-label');
    const scoresContainer = document.getElementById('scores-container');
    const explanationText = document.getElementById('explanation-text');
    const modelDetails = document.getElementById('model-details');
    const detailsGrid = document.getElementById('details-grid');
    const evalMetrics = document.getElementById('eval-metrics');
    const metricsContent = document.getElementById('metrics-content');
    const limitationsSection = document.getElementById('limitations-section');
    const limitationsList = document.getElementById('limitations-list');

    loadExamTypes();

    uploadArea.addEventListener('click', (e) => {
        if (e.target.id !== 'clear-upload') fileInput.click();
    });
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
    document.getElementById('clear-upload')?.addEventListener('click', (e) => { e.stopPropagation(); clearUpload(); });
    analyzeBtn.addEventListener('click', runAnalysis);
    resetBtn.addEventListener('click', resetAll);

    async function loadExamTypes() {
        try {
            const res = await fetch('/api/exam-types');
            const data = await res.json();
            examTypes = data.exam_types;
            renderExamTypes();
        } catch (err) {
            examContainer.innerHTML = '<p style="color: #ef4444;">Failed to load examination types.</p>';
        }
    }

    function renderExamTypes() {
        let html = '';
        for (const et of examTypes) {
            html += `
                <div class="exam-type-card" data-id="${et.id}" onclick="window._selectExam('${et.id}')">
                    <div class="exam-card-header">
                        <span class="exam-modality">${escapeHtml(et.modality)}</span>
                        <span class="exam-status-tag ready">● Ready & Verified</span>
                    </div>
                    <div class="exam-name">${escapeHtml(et.display_name)}</div>
                    <div class="exam-task">${escapeHtml(et.task_description)}</div>
                </div>
            `;
        }
        examContainer.innerHTML = html;
    }

    window._selectExam = function (id) {
        selectedExamType = id;
        document.querySelectorAll('.exam-type-card').forEach(c => c.classList.toggle('selected', c.dataset.id === id));
        step2.classList.remove('disabled');
        step3.classList.add('hidden');
        clearUpload();
        loadSamples(id);
    };

    async function loadSamples(examType) {
        try {
            const res = await fetch(`/api/samples/${examType}`);
            const data = await res.json();
            let html = '';
            for (const s of data.samples) {
                const isTumor = s.name.toLowerCase().includes('tumor');
                const isPneumonia = s.name.toLowerCase().includes('pneumonia');
                const tagClass = (isTumor || isPneumonia) ? 'tag-tumor' : 'tag-normal';
                let tagText = 'Normal Sample';
                if (isTumor) tagText = 'Tumor Sample';
                if (isPneumonia) tagText = 'Pneumonia Sample';

                html += `
                    <div class="sample-card" data-name="${s.name}" onclick="window._selectSample('${s.name}', this)">
                        <img src="${s.url}" alt="${s.name}">
                        <span class="sample-tag ${tagClass}">${tagText}</span>
                    </div>
                `;
            }
            samplesGrid.innerHTML = html;
        } catch (e) {
            samplesGrid.innerHTML = '<p style="color:#64748b;">No samples found.</p>';
        }
    }

    window._selectSample = function (name, el) {
        clearUpload();
        selectedSample = name;
        document.querySelectorAll('.sample-card').forEach(c => c.classList.toggle('selected', c.dataset.name === name));
        analyzeBtn.disabled = false;
    };

    function handleFile(file) {
        selectedFile = file;
        selectedSample = null;
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('selected'));
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.classList.add('hidden');
            uploadPreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
        analyzeBtn.disabled = false;
    }

    function clearUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadContent.classList.remove('hidden');
        uploadPreview.classList.add('hidden');
        previewImage.src = '';
        analyzeBtn.disabled = !selectedSample;
    }

    async function runAnalysis() {
        if (!selectedExamType) return;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<div class="spinner"></div> Running inference...';

        try {
            let res;
            if (selectedFile) {
                const fd = new FormData();
                fd.append('exam_type', selectedExamType);
                fd.append('image', selectedFile);
                res = await fetch('/api/predict', { method: 'POST', body: fd });
            } else {
                res = await fetch('/api/sample-predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ exam_type: selectedExamType, sample_name: selectedSample })
                });
            }
            const data = await res.json();
            displayResult(data, !res.ok);
        } catch (err) {
            displayResult({ error: 'Network communication failure.', is_supported: false }, true);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Run Research Inference';
        }
    }

    function displayResult(data, isError) {
        step3.classList.remove('hidden');
        step3.scrollIntoView({ behavior: 'smooth', block: 'start' });

        if (isError || !data.is_supported) {
            resultError.classList.remove('hidden');
            resultSuccess.classList.add('hidden');
            errorMessage.textContent = data.error || data.error_message || 'Inference error.';
            return;
        }

        resultError.classList.add('hidden');
        resultSuccess.classList.remove('hidden');

        const label = data.predicted_label;
        const isPositive = label === 'Tumor' || label === 'PNEUMONIA';
        predictionLabel.className = 'prediction-badge ' + (isPositive ? 'badge-tumor' : 'badge-normal');
        predictionLabel.textContent = label;

        scoresContainer.innerHTML = '';
        if (data.model_scores) {
            for (const [cls, score] of Object.entries(data.model_scores)) {
                const pct = Math.round(score * 100);
                scoresContainer.innerHTML += `
                    <div class="score-bar">
                        <span class="score-label">${escapeHtml(cls)}</span>
                        <div class="score-track">
                            <div class="score-fill" style="width: ${pct}%"></div>
                        </div>
                        <span class="score-value">${(score).toFixed(4)}</span>
                    </div>
                `;
            }
        }

        explanationText.textContent = data.explanation || '';

        // Details
        if (data.metadata) {
            showDetails(data.metadata);
            showMetrics(data.metadata);
            showLimitations(data.metadata);
        }
    }

    function showDetails(meta) {
        modelDetails.classList.remove('hidden');
        const fields = [
            ['Model Name', meta.name],
            ['Architecture', meta.architecture],
            ['Dataset', meta.dataset_name],
            ['Source / License', `${meta.dataset_source} (${meta.dataset_license})`],
            ['Task', meta.task]
        ];
        detailsGrid.innerHTML = fields.map(([k, v]) => `
            <div class="detail-row"><span class="detail-key">${k}</span><span class="detail-value">${escapeHtml(v || '')}</span></div>
        `).join('');
    }

    function showMetrics(meta) {
        if (!meta.evaluation_metrics) { evalMetrics.classList.add('hidden'); return; }
        evalMetrics.classList.remove('hidden');
        const m = meta.evaluation_metrics;
        let html = '<table class="metrics-table"><thead><tr><th>Metric</th><th>Score</th></tr></thead><tbody>';
        for (const k of ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']) {
            if (m[k] !== undefined) {
                html += `<tr><td>${k.toUpperCase()}</td><td>${Number(m[k]).toFixed(4)}</td></tr>`;
            }
        }
        html += '</tbody></table>';
        metricsContent.innerHTML = html;
    }

    function showLimitations(meta) {
        if (!meta.limitations?.length) return;
        limitationsSection.classList.remove('hidden');
        limitationsList.innerHTML = meta.limitations.map(l => `<li>${escapeHtml(l)}</li>`).join('');
    }

    function resetAll() {
        step2.classList.add('disabled');
        step3.classList.add('hidden');
        document.querySelectorAll('.exam-type-card').forEach(c => c.classList.remove('selected'));
        clearUpload();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }
})();
