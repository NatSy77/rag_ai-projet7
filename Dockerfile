# Image Python légère et compatible avec le projet
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Évite la création de fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Affiche immédiatement les logs Python
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système utiles
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copie du code de l'application
COPY app/ app/
COPY scripts/ scripts/

# Copie uniquement des données nécessaires à /ask
COPY data/event_chunks.csv data/event_chunks.csv
COPY data/chunks.index data/chunks.index

# Port utilisé par FastAPI/Uvicorn
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]