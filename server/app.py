"""
Serveur Flask d'Atlas.
Expose :
  GET  /        → Interface web de chat
  POST /chat    → Endpoint de conversation  {"message": "..."} → {"response": "..."}
  GET  /health  → Healthcheck pour Pterodactyl
"""

import json
import os
import random

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from brain.neural_network import ReseauNeuronal
from brain.tokenizer import Tokenizer

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

# Seuil minimal de confiance pour répondre
SEUIL_CONFIANCE = 0.4


def charger_modele():
    """
    Charge le modèle entraîné, le vocabulaire, les classes et les intents.
    Retourne un tuple (reseau, tokenizer, classes, intents_map) ou None si non disponible.
    """
    for chemin in [CHEMIN_MODELE, CHEMIN_VOCAB, CHEMIN_CLASSES, CHEMIN_INTENTS]:
        if not os.path.exists(chemin):
            return None

    # Charger le tokenizer
    tokenizer = Tokenizer()
    tokenizer.charger(CHEMIN_VOCAB)

    # Charger les classes
    with open(CHEMIN_CLASSES, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # Charger les intents (pour les réponses)
    with open(CHEMIN_INTENTS, "r", encoding="utf-8") as f:
        donnees = json.load(f)
    intents_map = {intent["tag"]: intent["responses"] for intent in donnees["intents"]}

    # Charger le réseau de neurones
    # On crée le réseau avec des dimensions temporaires (elles seront écrasées par charger())
    reseau = ReseauNeuronal(1, 1, 1, 1)
    reseau.charger(CHEMIN_MODELE)

    return reseau, tokenizer, classes, intents_map


# Chargement global au démarrage
modele_charge = charger_modele()

if modele_charge:
    reseau, tokenizer, classes, intents_map = modele_charge
    print("✅ Atlas est prêt à discuter !")
else:
    reseau = tokenizer = classes = intents_map = None
    print("⚠️  Modèle non trouvé. Lance 'python train.py' d'abord.")


# ─────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────

@app.route("/")
def index():
    """Sert l'interface web de chat."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint de conversation.
    Reçoit : {"message": "..."}
    Retourne : {"response": "..."}
    """
    if reseau is None:
        return jsonify({
            "response": "⚠️ Je ne suis pas encore entraîné ! Lance 'python train.py' pour m'entraîner."
        }), 503

    donnees = request.get_json(silent=True)
    if not donnees or "message" not in donnees:
        return jsonify({"error": "Paramètre 'message' manquant"}), 400

    message = donnees["message"].strip()
    if not message:
        return jsonify({"error": "Le message est vide"}), 400

    # Vectoriser le message
    vecteur = np.array([tokenizer.vectoriser(message)])

    # Prédire l'intent
    indice, confiance = reseau.predire(vecteur)

    # Vérifier le seuil de confiance
    if confiance < SEUIL_CONFIANCE:
        reponse = random.choice([
            "Je ne suis pas sûr de comprendre ta question. Peux-tu reformuler ? 🤔",
            "Hmm, je n'ai pas bien saisi. Tu peux préciser ? 😊",
            "Je suis encore en apprentissage ! Essaie de poser la question autrement.",
            "Je n'ai pas de réponse précise à ça... Tu peux essayer autrement ?"
        ])
    else:
        tag = classes[indice]
        reponses = intents_map.get(tag, ["Je n'ai pas de réponse à ça."])
        reponse = random.choice(reponses)

    return jsonify({"response": reponse})


@app.route("/health")
def health():
    """Healthcheck pour Pterodactyl."""
    return jsonify({
        "status": "ok",
        "modele_charge": reseau is not None,
        "vocabulaire": len(tokenizer.vocabulaire) if tokenizer else 0
    })


# ─────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("HOST", "163.5.59.154")
    port = int(os.environ.get("PORT", 7778))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Atlas démarre sur {host}:{port}...")
    app.run(host=host, port=port, debug=debug)
