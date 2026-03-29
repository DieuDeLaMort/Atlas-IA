#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Script de démarrage d'Atlas
# 1. Vérifie si le modèle est entraîné
# 2. Si non, lance l'entraînement automatiquement
# 3. Lance le serveur Flask
# ─────────────────────────────────────────────────────────────────────────────

set -e

MODELE="brain/model.json"

echo "=============================================="
echo "      🤖 Atlas — Démarrage"
echo "=============================================="

# Vérifier si le modèle entraîné existe
if [ ! -f "$MODELE" ]; then
    echo "⚠️  Modèle non trouvé. Lancement de l'entraînement..."
    echo ""
    python train.py
    echo ""
    echo "✅ Entraînement terminé !"
else
    echo "✅ Modèle trouvé : $MODELE"
fi

echo ""
echo "🚀 Lancement du serveur Flask..."
echo "=============================================="

# Lancer le serveur Flask
python -m server.app
