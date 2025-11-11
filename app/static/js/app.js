class LoanPredictor {
    constructor() {
        this.form = document.getElementById('predictionForm');
        this.resultSection = document.getElementById('resultSection');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.resetBtn = document.getElementById('resetBtn');

        this.initEventListeners();
    }

    initEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.resetBtn.addEventListener('click', () => this.resetForm());

        // Calcolo automatico del reddito totale
        document.getElementById('ApplicantIncome').addEventListener('input', () => this.calculateTotalIncome());
        document.getElementById('CoapplicantIncome').addEventListener('input', () => this.calculateTotalIncome());
    }

    calculateTotalIncome() {
        const applicantIncome = parseFloat(document.getElementById('ApplicantIncome').value) || 0;
        const coapplicantIncome = parseFloat(document.getElementById('CoapplicantIncome').value) || 0;
        const totalIncome = applicantIncome + coapplicantIncome;

        // Puoi mostrare il totale se vuoi, ma non lo inviamo direttamente
        console.log('Reddito totale calcolato:', totalIncome);
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
        const applicantIncome = parseFloat(document.getElementById('ApplicantIncome').value);
        const coapplicantIncome = parseFloat(document.getElementById('CoapplicantIncome').value);
        const totalIncome = applicantIncome + coapplicantIncome;
        const loanAmount = parseFloat(formData.get('LoanAmount'));

        // Calcolo delle feature derivate
        const loanAmountToIncome = loanAmount / totalIncome;
        const hasDependents = parseInt(formData.get('Dependents')) > 0 ? 1 : 0;

        // Codifica one-hot per Property_Area
        const propertyArea = formData.get('Property_Area');
        const propertyAreaSemiurban = propertyArea === 'Semiurban' ? 1 : 0;
        const propertyAreaUrban = propertyArea === 'Urban' ? 1 : 0;

        // Codifica per le feature categoriche
        const isMarried = parseInt(formData.get('Married_Yes'));
        const isGraduate = parseInt(formData.get('Education_Not Graduate')) === 0 ? 1 : 0;

        // Calcolo income bracket (esempio: soglia a 5000)
        const incomeBracket = totalIncome > 5000 ? 1 : 0;
        const creditXIncomeHigh = parseInt(formData.get('Credit_History')) * incomeBracket;

        return [
            parseFloat(formData.get('Dependents')),      // Dependents
            loanAmount,                                  // LoanAmount
            parseFloat(formData.get('Loan_Amount_Term')), // Loan_Amount_Term
            parseFloat(formData.get('Credit_History')),  // Credit_History
            totalIncome,                                 // TotalIncome
            loanAmountToIncome,                          // LoanAmount_to_Income
            hasDependents,                               // HasDependents
            isMarried,                                   // IsMarried
            isGraduate,                                  // IsGraduate
            incomeBracket,                               // Income_bracket
            creditXIncomeHigh,                           // Credit_x_IncomeHigh
            parseFloat(formData.get('Gender_Male')),     // Gender_Male
            isMarried,                                   // Married_Yes (duplicato per compatibilità)
            parseInt(formData.get('Education_Not Graduate')), // Education_Not Graduate
            parseInt(formData.get('Self_Employed_Yes')), // Self_Employed_Yes
            propertyAreaSemiurban,                       // Property_Area_Semiurban
            propertyAreaUrban                            // Property_Area_Urban
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
        const isApproved = prediction.predicted_class === 1;
        const approvedProb = (prediction.probabilities[1] * 100).toFixed(1);
        const rejectedProb = (prediction.probabilities[0] * 100).toFixed(1);

        // Aggiorna il risultato principale
        document.getElementById('predictionResult').textContent =
            isApproved ? '✅ PRESTITO APPROVATO' : '❌ PRESTITO RIFIUTATO';
        document.getElementById('predictionResult').className =
            `display-6 ${isApproved ? 'text-success' : 'text-danger'}`;

        // Aggiorna le probabilità
        document.getElementById('probabilityValue').textContent = `${approvedProb}%`;
        document.getElementById('probRejected').textContent = `${rejectedProb}%`;
        document.getElementById('probApproved').textContent = `${approvedProb}%`;

        // Anima la barra delle probabilità
        this.animateProbabilityBar(approvedProb);

        // Mostra i fattori chiave
        this.displayKeyFactors(prediction, isApproved);

        // Mostra la sezione risultati
        this.resultSection.style.display = 'block';
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    animateProbabilityBar(percentage) {
        const bar = document.querySelector('.probability-bar');
        bar.style.width = '0%';

        setTimeout(() => {
            bar.style.transition = 'width 1.5s ease-in-out';
            bar.style.width = `${percentage}%`;
        }, 100);
    }

    displayKeyFactors(prediction, isApproved) {
        const factorsDiv = document.getElementById('keyFactors');
        let factorsHTML = '';

        if (isApproved) {
            factorsHTML = `
                <strong>Fattori positivi:</strong>
                <ul class="mb-0 mt-2">
                    <li>Storia creditizia positiva</li>
                    <li>Rapporto prestito/reddito favorevole</li>
                    <li>Stabilità finanziaria adeguata</li>
                </ul>
            `;
        } else {
            factorsHTML = `
                <strong>Aree di miglioramento:</strong>
                <ul class="mb-0 mt-2">
                    <li>Migliorare la storia creditizia</li>
                    <li>Ridurre il rapporto prestito/reddito</li>
                    <li>Aumentare la stabilità finanziaria</li>
                </ul>
            `;
        }

        factorsDiv.innerHTML = factorsHTML;
    }

    displayError(error) {
        alert(`Errore: ${error.message}`);
        console.error('Prediction error:', error);
    }

    showLoading() {
        this.loadingSpinner.style.display = 'block';
        this.form.querySelector('button').disabled = true;
    }

    hideLoading() {
        this.loadingSpinner.style.display = 'none';
        this.form.querySelector('button').disabled = false;
    }

    resetForm() {
        this.form.reset();
        this.resultSection.style.display = 'none';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Inizializza l'app quando il DOM è caricato
document.addEventListener('DOMContentLoaded', () => {
    new LoanPredictor();
});