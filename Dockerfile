# Usa un'immagine base di Python
FROM python:3.10-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file requirements.txt nel container
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutti i file del progetto nella directory di lavoro del container
COPY . .

# Comando da eseguire quando parte il container
# Assumendo che il tuo script principale aggiornato si chiami main.py
CMD ["python", "main.py"]

# Se invece vuoi usare Streamlit:
# EXPOSE 8501
# CMD ["streamlit", "run", "interactive_model.py", "--server.port=8501", "--server.address=0.0.0.0"]