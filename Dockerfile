# Dockerfile
# Usa un'immagine base leggera con Python 3.10
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia solo requirements per sfruttare la cache Docker
COPY requirements.txt .

# Aggiorna pip e installa le dipendenze
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia tutto il progetto nella directory /app
COPY . .

# Imposta PYTHONPATH per permettere import come "from classes import ..."
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Espone la porta del server Flask
EXPOSE 5000

# Comando di avvio: lancia l’API Flask
CMD ["python", "app/app.py"]