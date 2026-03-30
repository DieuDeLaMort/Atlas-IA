FROM python:3.10-slim

# Répertoire de travail
WORKDIR /app

# Copier les dépendances en premier (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Port exposé (configurable via variable d'environnement)
ENV PORT=5000
EXPOSE 5000

# Rendre le script de démarrage exécutable (fallback)
RUN chmod +x start.sh

# Démarrage via main.py
CMD ["python", "main.py"]
