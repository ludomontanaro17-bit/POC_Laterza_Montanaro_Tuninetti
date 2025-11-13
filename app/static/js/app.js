class LoanPredictor {
    constructor() {
        this.form = document.getElementById('predictionForm');
        this.resultSection = document.getElementById('resultSection');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.resetBtn = document.getElementById('resetBtn');
        this.finalPrediction = null;

        this.modelNames = {
            'keras': 'Rete Neurale',
            'logreg': 'Regressione Logistica',
            'xgboost': 'XGBoost'
        };

        this.modelColors = {
            'keras': '#e74c3c',
            'logreg': '#3498db',
            'xgboost': '#9b59b6'
        };

        this.initEventListeners();
    }

    initEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.resetBtn.addEventListener('click', () => this.resetForm());

        // Calcolo automatico del reddito totale
        document.getElementById('ApplicantIncome').addEventListener('input', () => this.calculateTotalIncome());
        document.getElementById('CoapplicantIncome').addEventListener('input', () => this.calculateTotalIncome());

        // Inizializza il calcolo del reddito totale
        this.calculateTotalIncome();
    }

    calculateTotalIncome() {
        const applicantIncome = parseFloat(document.getElementById('ApplicantIncome').value) || 0;
        const coapplicantIncome = parseFloat(document.getElementById('CoapplicantIncome').value) || 0;
        const totalIncome = applicantIncome + coapplicantIncome;

        document.getElementById('totalIncomeDisplay').textContent = `€${totalIncome.toLocaleString()}`;
    }

    async handleSubmit(e) {
        e.preventDefault();

        this.showLoading();

        try {
            const features = this.prepareFeatures();
            const prediction = await this.makePrediction(features);
            this.displayResults(prediction);
        } catch (error) {
            this.displayError(error);
        } finally {
            this.hideLoading();
        }
    }

    prepareFeatures() {
        const formData = new FormData(this.form);

        // 🔥 ORDINE ESATTO DELLE FEATURE COME NEL MODELLO 🔥
        // 0: "Gender"
        // 1: "Married"
        // 2: "Dependents"
        // 3: "Education"
        // 4: "Self_Employed"
        // 5: "LoanAmount"
        // 6: "Loan_Amount_Term"
        // 7: "Credit_History"
        // 8: "TotalIncome"
        // 9: "Property_Area_Semiurban"
        // 10: "const formData = new FormData(this.form);

        // 🔥 ORDINE ESATTO DELLE FEATURE COME NEL MODELLO 🔥
        // 0: "Gender"
        // 1: "Married"
        // 2: "Dependents"
        // 3: "Education"
        // 4: "Self_Employed"
        // 5: "LoanAmount"
        // 6: "Loan_Amount_Term"
        // 7: "CProperty_Area_Urban"

        // Calcolo delle feature di base
        const applicantIncome = parseFloat(document.getElementById('ApplicantIncome').value) || 0;
        const coapplicantIncome = parseFloat(document.getElementById('CoapplicantIncome').value) || 0;
        const totalIncome = applicantIncome + coapplicantIncome;

        // Prepara i valori per le feature categoriche
        const gender = formData.get('Gender') === 'Male' ? 1 : 0;
        const married = formData.get('Married') === 'Yes' ? 1 : 0;
        const education = formData.get('Education') === 'Graduate' ? 1 : 0;
        const selfEmployed = formData.get('Self_Employed') === 'Yes' ? 1 : 0;

        // Prepara Property_Area (one-hot encoding)
        const propertyArea = formData.get('Property_Area');
        const propertyAreaSemiurban = propertyArea === 'Semiurban' ? 1 : 0;
        const propertyAreaUrban = propertyArea === 'Urban' ? 1 : 0;

        // 🔥 ARRAY NELL'ORDINE ESATTO ATTESO DAL MODELLO 🔥
        return [
            gender,                         // 0: "Gender" (Male=1, Female=0)
            married,                        // 1: "Married" (Yes=1, No=0)
            parseFloat(formData.get('Dependents')) || 0, // 2: "Dependents" (0,1,2,3)
            education,                      // 3: "Education" (Graduate=1, Not Graduate=0)
            selfEmployed,                   // 4: "Self_Employed" (Yes=1, No=0)
            parseFloat(formData.get('LoanAmount')) || 0, // 5: "LoanAmount"
            parseFloat(formData.get('Loan_Amount_Term')) || 0, // 6: "Loan_Amount_Term"
            parseFloat(formData.get('Credit_History')) || 0, // 7: "Credit_History" (1=Yes, 0=No)
            totalIncome,                    // 8: "TotalIncome"
            propertyAreaSemiurban,          // 9: "Property_Area_Semiurban"
            propertyAreaUrban               // 10: "Property_Area_Urban"
        ];
    }


    async makePrediction(features) {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: features
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Errore nella predizione');
        }

        return await response.json();
    }

    displayResults(prediction) {
        this.finalPrediction = prediction.final_prediction;
        const isApproved = this.finalPrediction === 1;
        const approvedProb = (prediction.final_probability * 100).toFixed(1);
        const rejectedProb = ((1 - prediction.final_probability) * 100).toFixed(1);

        // Aggiorna il risultato principale
        const resultElement = document.getElementById('predictionResult');
        resultElement.textContent = isApproved ?
            '✅ PRESTITO APPROVATO' : '❌ PRESTITO RIFIUTATO';
        resultElement.className = `display-5 fw-bold mb-3 ${isApproved ? 'text-success' : 'text-danger'}`;

        // Aggiorna la sezione risultati
        this.resultSection.className = `result-section ${isApproved ? 'result-approved' : 'result-rejected'}`;

        // Aggiorna le probabilità
        document.getElementById('probRejected').textContent = `${rejectedProb}%`;
        document.getElementById('probApproved').textContent = `${approvedProb}%`;
        document.getElementById('finalProbabilityDisplay').textContent = `${approvedProb}%`;

        // Anima la barra delle probabilità
        this.animateProbabilityBar(approvedProb);

        // Mostra i dettagli dell'ensemble
        this.displayEnsembleDetails(prediction);

        // Mostra la sezione risultati
        this.resultSection.style.display = 'block';
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    animateProbabilityBar(percentage) {
        const bar = document.getElementById('probabilityBar');
        const value = document.getElementById('probabilityValue');

        // Reset per l'animazione
        bar.style.width = '0%';
        value.textContent = '0%';

        setTimeout(() => {
            bar.style.transition = 'width 2s cubic-bezier(0.4, 0, 0.2, 1)';
            bar.style.width = `${percentage}%`;

            // Animazione del contatore - VERSIONE CORRETTA
            this.animateValue(0, parseFloat(percentage), 2000, (currentValue) => {
                value.textContent = `${Math.round(currentValue)}%`;
            });
        }, 300);
    }

    animateValue(start, end, duration, callback) {
        const startTime = performance.now();
        const change = end - start;

        const updateValue = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentValue = start + (change * easeOutQuart);

            callback(currentValue);

            if (progress < 1) {
                requestAnimationFrame(updateValue);
            }
        }

        requestAnimationFrame(updateValue);
    }

    animateValue(start, end, duration, callback, element) {
        const startTime = performance.now();
        const change = end - start;

        function updateValue(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentValue = start + (change * easeOutQuart);

            callback(currentValue, element);

            if (progress < 1) {
                requestAnimationFrame(updateValue);
            }
        }

        requestAnimationFrame(updateValue);
    }

    displayEnsembleDetails(prediction) {
        this.displayModelPredictions(prediction.individual_predictions);
        this.displayVotingDetails(prediction.ensemble_details, prediction.final_probability);
        this.displayKeyFactors(prediction, this.finalPrediction === 1);
    }

    displayModelPredictions(individualPredictions) {
        const container = document.getElementById('modelPredictions');
        let html = '';

        for (const [modelKey, probability] of Object.entries(individualPredictions)) {
            const modelName = this.modelNames[modelKey] || modelKey;
            const percentage = (probability * 100).toFixed(1);
            const barWidth = probability * 100;
            const confidence = this.getConfidenceLevel(probability);
            const modelColor = this.modelColors[modelKey] || '#95a5a6';

            html += `
                <div class="model-card">
                    <div class="row align-items-center">
                        <div class="col-md-3">
                            <div class="d-flex align-items-center">
                                <div class="model-icon me-3" style="color: ${modelColor};">
                                    <i class="fas fa-${this.getModelIcon(modelKey)} fa-2x"></i>
                                </div>
                                <div>
                                    <h6 class="mb-0">${modelName}</h6>
                                    <span class="badge confidence-badge" style="background-color: ${modelColor}">
                                        ${confidence}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="progress" style="height: 12px; border-radius: 6px;">
                                <div class="progress-bar" 
                                     style="width: ${barWidth}%; background-color: ${modelColor};" 
                                     role="progressbar">
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3 text-end">
                            <span class="fw-bold fs-5" style="color: ${modelColor};">${percentage}%</span>
                        </div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    displayVotingDetails(votingDetails, finalProbability) {
        const container = document.getElementById('votingDetails');
        let html = '';

        // Intestazione della tabella
        html += `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead class="table-dark">
                        <tr>
                            <th>Modello</th>
                            <th>Probabilità</th>
                            <th>Peso</th>
                            <th>Contributo</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        // Righe dei modelli
        votingDetails.forEach(detail => {
            const modelName = this.modelNames[detail.model] || detail.model;
            const probPercentage = (detail.probability * 100).toFixed(1);
            const weightedPercentage = (detail.weighted_prob * 100).toFixed(1);
            const weightPercentage = (detail.weight * 100).toFixed(0);
            const modelColor = this.modelColors[detail.model] || '#95a5a6';

            html += `
                <tr>
                    <td>
                        <i class="fas fa-${this.getModelIcon(detail.model)} me-2" style="color: ${modelColor};"></i>
                        ${modelName}
                    </td>
                    <td>${probPercentage}%</td>
                    <td>
                        <span class="badge bg-secondary">${weightPercentage}%</span>
                    </td>
                    <td class="fw-bold" style="color: ${modelColor};">${weightedPercentage}%</td>
                </tr>
            `;
        });

        // Riga finale
        const finalPercentage = (finalProbability * 100).toFixed(1);
        html += `
                    </tbody>
                    <tfoot class="table-success">
                        <tr>
                            <td colspan="3" class="text-end fw-bold">Probabilità Finale:</td>
                            <td class="fw-bold fs-5">${finalPercentage}%</td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        `;

        container.innerHTML = html;
    }

    displayKeyFactors(prediction, isApproved) {
        const factorsDiv = document.getElementById('keyFactors');
        const modelAgreement = this.analyzeModelAgreement(prediction.individual_predictions);

        let factorsHTML = '';

        if (isApproved) {
            factorsHTML = `
                <div class="alert alert-success">
                    <h6><i class="fas fa-thumbs-up me-2"></i>Analisi Positiva</h6>
                    <ul class="mb-0">
                        <li>Consenso tra modelli: <strong>${modelAgreement.agreementLevel}</strong></li>
                        <li>${modelAgreement.confidentModels} modelli mostrano alta confidenza</li>
                        <li>Storia creditizia positiva rilevata</li>
                        <li>Rapporto prestito/reddito nella norma</li>
                    </ul>
                </div>
            `;
        } else {
            factorsHTML = `
                <div class="alert alert-warning">
                    <h6><i class="fas fa-exclamation-triangle me-2"></i>Aree di Attenzione</h6>
                    <ul class="mb-0">
                        <li>Consenso tra modelli: <strong>${modelAgreement.agreementLevel}</strong></li>
                        <li>${modelAgreement.confidentModels} modelli mostrano alta confidenza</li>
                        ${modelAgreement.divergentModels.length > 0 ?
                    `<li>Modelli in disaccordo: ${modelAgreement.divergentModels.map(m => this.modelNames[m]).join(', ')}</li>` : ''}
                        <li>Si consiglia di verificare la storia creditizia</li>
                    </ul>
                </div>
            `;
        }

        factorsDiv.innerHTML = factorsHTML;
    }

    getModelIcon(modelKey) {
        const icons = {
            'keras': 'brain',
            'logreg': 'chart-line',
            'xgboost': 'rocket'
        };
        return icons[modelKey] || 'microchip';
    }

    getConfidenceLevel(probability) {
        const confidence = Math.abs(probability - 0.5) * 2; // 0-1 scale
        if (confidence > 0.7) return 'Alta';
        if (confidence > 0.4) return 'Media';
        return 'Bassa';
    }

    analyzeModelAgreement(individualPredictions) {
        const threshold = 0.6;
        let confidentModels = 0;
        const divergentModels = [];

        for (const [model, prob] of Object.entries(individualPredictions)) {
            if (prob > threshold || prob < (1 - threshold)) {
                confidentModels++;
            }
            if ((prob > 0.5) !== (this.finalPrediction > 0.5)) {
                divergentModels.push(model);
            }
        }

        const totalModels = Object.keys(individualPredictions).length;
        let agreementLevel;

        if (confidentModels === totalModels) agreementLevel = 'Alto';
        else if (confidentModels >= Math.ceil(totalModels / 2)) agreementLevel = 'Medio';
        else agreementLevel = 'Basso';

        return {
            agreementLevel,
            confidentModels,
            divergentModels
        };
    }

    displayError(error) {
        // Mostra un alert più elegante
        const errorHTML = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <h4 class="alert-heading"><i class="fas fa-exclamation-triangle me-2"></i>Errore</h4>
                <p class="mb-0">${error.message}</p>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        // Inserisce l'alert all'inizio del container principale
        const mainCard = document.querySelector('.glass-card');
        mainCard.insertAdjacentHTML('afterbegin', errorHTML);

        console.error('Prediction error:', error);
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.form.querySelector('button').disabled = true;
        this.form.querySelector('button').innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>ANALISI IN CORSO...';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        this.form.querySelector('button').disabled = false;
        this.form.querySelector('button').innerHTML = '<i class="fas fa-brain me-2"></i>ANALIZZA CON ENSEMBLE AI';
    }

    resetForm() {
        this.form.reset();
        this.resultSection.style.display = 'none';
        this.calculateTotalIncome(); // Aggiorna il display del reddito totale

        // Rimuovi eventuali alert di errore
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => alert.remove());

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Inizializza l'app quando il DOM è caricato
document.addEventListener('DOMContentLoaded', () => {
    new LoanPredictor();

    // Aggiungi alcuni dati di esempio per testing
    document.getElementById('ApplicantIncome').value = 5000;
    document.getElementById('CoapplicantIncome').value = 2000;
    document.getElementsByName('LoanAmount')[0].value = 150000;
    document.getElementsByName('Loan_Amount_Term')[0].value = 360;
    document.getElementsByName('Dependents')[0].value = 1;

    // Aggiorna il calcolo del reddito totale
    const predictor = new LoanPredictor();
    predictor.calculateTotalIncome();
});