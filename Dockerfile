FROM python:3.10-slim

# Répertoire de travail
WORKDIR /app

# Copier les dépendances en premier (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Host et port de connexion (configurable via variables d'environnement)
# En Docker, on écoute sur 0.0.0.0 (Docker gère le réseau) ; le port cible est 7778
ENV HOST=0.0.0.0
ENV PORT=7778
EXPOSE 7778

# Rendre le script de démarrage exécutable (fallback)
RUN chmod +x start.sh

# Démarrage via main.py
CMD ["python", "main.py"]
