"""
Point d'entrée principal d'Atlas.
Lance : python main.py

1. Vérifie si le modèle est entraîné
2. Si non, lance l'entraînement automatiquement
3. Lance le serveur Flask

Un système de logs enregistre les erreurs et crashs dans logs/atlas.log.
"""

import logging
import os
import sys
import traceback

# ─────────────────────────────────────────────────
# Configuration du logging
# ─────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "atlas.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("atlas")

CHEMIN_MODELE = os.path.join("brain", "model.json")


# ─────────────────────────────────────────────────
# Détection robuste de l'IP et du port
# ─────────────────────────────────────────────────


def detecter_host_port():
    """
    Détecte l'IP et le port de manière robuste.
    Pterodactyl définit SERVER_IP et SERVER_PORT.
    Fallback sur HOST/PORT, puis 0.0.0.0:7778.
    Ne crash jamais : retourne les valeurs par défaut en cas de problème.
    """
    host = os.environ.get("SERVER_IP") or os.environ.get("HOST", "0.0.0.0")

    port_str = os.environ.get("SERVER_PORT") or os.environ.get("PORT", "7778")
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f"Port hors limites : {port}")
    except (ValueError, TypeError) as e:
        logger.warning("Port invalide '%s' (%s). Utilisation du port par défaut 7778.", port_str, e)
        port = 7778

    return host, port


# ─────────────────────────────────────────────────
# Entraînement du modèle
# ─────────────────────────────────────────────────


def entrainer_modele():
    """Lance l'entraînement si le modèle n'existe pas. Retourne True si OK."""
    if os.path.isfile(CHEMIN_MODELE):
        # Vérifier que le fichier est lisible et non vide
        try:
            taille = os.path.getsize(CHEMIN_MODELE)
            if taille == 0:
                logger.warning("Fichier modèle vide (%s). Ré-entraînement nécessaire.", CHEMIN_MODELE)
            else:
                logger.info("Modèle trouvé : %s (%d octets)", CHEMIN_MODELE, taille)
                return True
        except OSError as e:
            logger.warning("Impossible de lire le modèle (%s) : %s", CHEMIN_MODELE, e)

    logger.info("Modèle non trouvé. Lancement de l'entraînement...")

    try:
        from brain.trainer import Trainer

        trainer = Trainer(
            chemin_intents="data/intents.json",
            chemin_modele="brain/model.json",
            chemin_vocab="data/vocabulary.json",
            taille_cachee1=128,
            taille_cachee2=64,
            taux_apprentissage=0.01,
            epochs=5000,
        )
        trainer.lancer()
        logger.info("Entraînement terminé avec succès !")
        return True
    except Exception:
        logger.error("Erreur lors de l'entraînement :\n%s", traceback.format_exc())
        return False


# ─────────────────────────────────────────────────
# Lancement du serveur
# ─────────────────────────────────────────────────


def lancer_serveur():
    """Démarre le serveur Flask. Retourne False en cas d'erreur fatale."""
    try:
        from server.app import app
    except Exception:
        logger.error("Impossible d'importer le serveur Flask :\n%s", traceback.format_exc())
        return False

    host, port = detecter_host_port()
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info("Atlas démarre sur %s:%d ...", host, port)

    try:
        app.run(host=host, port=port, debug=debug)
    except OSError as e:
        # Port déjà utilisé, permission refusée, etc.
        logger.error("Impossible de démarrer le serveur sur %s:%d — %s", host, port, e)
        return False
    except Exception:
        logger.error("Crash du serveur Flask :\n%s", traceback.format_exc())
        return False

    return True


# ─────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────


def main():
    logger.info("==============================================")
    logger.info("      🤖 Atlas — Démarrage")
    logger.info("==============================================")

    # Étape 1 : entraînement si nécessaire
    if not entrainer_modele():
        logger.error("Arrêt : l'entraînement a échoué.")
        return 1

    logger.info("🚀 Lancement du serveur Flask...")
    logger.info("==============================================")

    # Étape 2 : lancement du serveur
    if not lancer_serveur():
        logger.error("Arrêt : le serveur n'a pas pu démarrer.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur (Ctrl+C).")
        code = 0
    except Exception:
        logger.critical("Crash non géré :\n%s", traceback.format_exc())
        code = 1

    sys.exit(code)
