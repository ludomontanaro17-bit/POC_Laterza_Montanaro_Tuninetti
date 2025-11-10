# POC_Laterza_Montanaro_Tuninetti
Statistiche descrittive suggerite da includere (da calcolare con i dati):
Distribuzione delle classi target (percentuale approvati vs non approvati)
Conta e percentuale per categorie di Gender, Marital Status, Education, Property Area
Media, mediana, deviazione standard per Income e Loan Amount
Percentuale con Credit History positivo
Correlazioni tra variabili numeriche (Income, Loan Amount) e target
Cross-tab tra Credit History e Loan Status (forte predittore atteso)
Pulizia e preprocessing raccomandati

Gestire valori mancanti (imputazione per Income/Loan Amount, categoria “Unknown” per categorical)
Normalizzazione o scaling di Income e Loan Amount per modelli sensibili alla scala
Codifica delle categoriche (one-hot o target encoding per variabili ordinali)
Trasformazione dipendenti: convertire "3+" in 3 o in categoria separata
Verifica outlier su Income e Loan Amount
Feature engineering consigliata

Rapporto LoanAmount/Income (debito rispetto al reddito)
Indicatori binari: HasDependents, IsMarried, IsGraduate
Interazioni: CreditHistory × Income bracket
Binning Income in fasce (basso/medio/alto)