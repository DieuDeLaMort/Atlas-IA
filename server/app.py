"""
Serveur Flask d'Atlas v2.0.
Expose :
  GET  /           → Interface web Jarvis UI
  POST /chat       → Endpoint de conversation {"message": "..."} → {"response": "...", "confidence": 0.9}
  GET  /health     → Healthcheck pour Pterodactyl
  GET  /status     → Statut détaillé du système
  GET  /history    → Historique de la conversation en cours
  POST /history/clear → Efface l'historique
  GET  /version    → Version d'Atlas
"""

import json
import logging
import os
import random
import time
import traceback
from collections import deque

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from brain.neural_network import ReseauNeuronal
from brain.tokenizer import Tokenizer
from brain import web_search

logger = logging.getLogger("atlas.server")

ATLAS_VERSION = "2.0.0"

# ─────────────────────────────────────────────────
# Initialisation Flask
# ─────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────────

CHEMIN_MODELE = os.path.join(os.path.dirname(__file__), "..", "brain", "model.json")
CHEMIN_VOCAB = os.path.join(os.path.dirname(__file__), "..", "data", "vocabulary.json")
CHEMIN_CLASSES = os.path.join(os.path.dirname(__file__), "..", "brain", "classes.json")
CHEMIN_INTENTS = os.path.join(os.path.dirname(__file__), "..", "data", "intents.json")

# Seuil minimal de confiance pour répondre avec le modèle local
SEUIL_CONFIANCE = 0.35

# Historique de conversation en mémoire (max 100 échanges)
_historique = deque(maxlen=100)

# Timestamp du démarrage du serveur
_start_time = time.time()


def charger_modele():
    """
    Charge le modèle entraîné, le vocabulaire, les classes et les intents.
    Retourne un tuple (reseau, tokenizer, classes, intents_map) ou None si non disponible.
    """
    for chemin in [CHEMIN_MODELE, CHEMIN_VOCAB, CHEMIN_CLASSES, CHEMIN_INTENTS]:
        if not os.path.exists(chemin):
            logger.warning("Fichier manquant : %s", chemin)
            return None

    try:
        tokenizer = Tokenizer()
        tokenizer.charger(CHEMIN_VOCAB)

        with open(CHEMIN_CLASSES, "r", encoding="utf-8") as f:
            classes = json.load(f)

        with open(CHEMIN_INTENTS, "r", encoding="utf-8") as f:
            donnees = json.load(f)
        intents_map = {intent["tag"]: intent["responses"] for intent in donnees["intents"]}

        # Créer le réseau avec des dimensions temporaires (écrasées par charger())
        reseau = ReseauNeuronal(1, 1, 1, 1)
        reseau.charger(CHEMIN_MODELE)

        return reseau, tokenizer, classes, intents_map

    except Exception:
        logger.error("Erreur lors du chargement du modèle :\n%s", traceback.format_exc())
        return None


# Chargement global au démarrage
modele_charge = charger_modele()

if modele_charge:
    reseau, tokenizer, classes, intents_map = modele_charge
    logger.info("✅ Atlas v%s est prêt à discuter !", ATLAS_VERSION)
else:
    reseau = tokenizer = classes = intents_map = None
    logger.warning("⚠️  Modèle non trouvé. Lance 'python main.py' d'abord.")


# ─────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────

@app.route("/")
def index():
    """Sert l'interface web Jarvis UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint de conversation.
    Reçoit : {"message": "..."}
    Retourne : {"response": "...", "confidence": 0.95, "source": "model|web|fallback"}
    """
    donnees = request.get_json(silent=True)
    if not donnees or "message" not in donnees:
        return jsonify({"error": "Paramètre 'message' manquant"}), 400

    message = donnees["message"].strip()
    if not message:
        return jsonify({"error": "Le message est vide"}), 400

    try:
        reponse_locale = None
        confiance = 0.0
        source = "fallback"

        # ── Modèle local (si disponible) ──
        if reseau is not None:
            vecteur = np.array([tokenizer.vectoriser(message)])
            indice, confiance = reseau.predire(vecteur)

            if confiance >= SEUIL_CONFIANCE:
                tag = classes[indice]
                reponses = intents_map.get(tag, [])
                if reponses:
                    reponse_locale = random.choice(reponses)
                    source = "model"

        # ── Recherche web si le modèle local n'a pas de réponse confiante ──
        if reponse_locale is None:
            logger.info("Recherche web pour : %s", message)
            reponse_web = web_search.chercher(message)
            if reponse_web:
                reponse = f"🌐 {reponse_web}"
                source = "web"
            else:
                reponse = random.choice([
                    "Je n'ai trouvé aucune information sur ce sujet. Essaie de reformuler ! 🤔",
                    "Hmm, je n'ai rien trouvé. Tu peux préciser ta question ? 😊",
                    "Aucune source disponible pour ça. Essaie autrement !",
                ])
                source = "fallback"
        else:
            reponse = reponse_locale

        # Stocker dans l'historique
        _historique.append({
            "role": "user",
            "content": message,
            "timestamp": time.time(),
        })
        _historique.append({
            "role": "atlas",
            "content": reponse,
            "confidence": round(confiance, 4),
            "source": source,
            "timestamp": time.time(),
        })

        return jsonify({
            "response": reponse,
            "confidence": round(confiance, 4),
            "source": source,
        })

    except Exception:
        logger.error("Erreur lors du traitement du message :\n%s", traceback.format_exc())
        return jsonify({"response": "⚠️ Une erreur interne est survenue. Réessaie !"}), 500


@app.route("/health")
def health():
    """Healthcheck pour Pterodactyl."""
    return jsonify({
        "status": "ok",
        "modele_charge": reseau is not None,
        "vocabulaire": len(tokenizer.vocabulaire) if tokenizer else 0,
    })


@app.route("/status")
def status():
    """Statut détaillé du système Atlas."""
    uptime = int(time.time() - _start_time)
    nb_intents = len(classes) if classes else 0
    nb_vocab = len(tokenizer.vocabulaire) if tokenizer else 0

    arch = None
    if reseau is not None:
        arch = {
            "entree": reseau.taille_entree,
            "cachee1": reseau.taille_cachee1,
            "cachee2": reseau.taille_cachee2,
            "cachee3": reseau.taille_cachee3,
            "sortie": reseau.taille_sortie,
            "dropout": reseau.taux_dropout,
            "optimiseur": "Adam",
        }

    return jsonify({
        "version": ATLAS_VERSION,
        "status": "online",
        "modele_charge": reseau is not None,
        "nb_intents": nb_intents,
        "vocabulaire": nb_vocab,
        "historique": len(_historique),
        "uptime_secondes": uptime,
        "seuil_confiance": SEUIL_CONFIANCE,
        "architecture": arch,
    })


@app.route("/version")
def version():
    """Retourne la version d'Atlas."""
    return jsonify({"version": ATLAS_VERSION, "nom": "Atlas IA"})


@app.route("/history")
def history():
    """Retourne l'historique de la conversation en cours."""
    limit = request.args.get("limit", 50, type=int)
    items = list(_historique)[-limit:]
    return jsonify({"history": items, "total": len(_historique)})


@app.route("/history/clear", methods=["POST"])
def history_clear():
    """Efface l'historique de la conversation."""
    _historique.clear()
    return jsonify({"status": "ok", "message": "Historique effacé."})


# ─────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.exit(
        "L'exécution directe de server/app.py n'est pas supportée.\n"
        "Utilise 'python main.py' pour démarrer Atlas avec le logging et la gestion d'erreurs."
    )
