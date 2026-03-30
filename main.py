"""
Point d'entrée principal d'Atlas v2.0.
Lance : python main.py

1. Affiche la bannière de démarrage
2. Vérifie si le modèle est à jour par rapport aux intents
3. Lance l'entraînement automatiquement si nécessaire
4. Lance le serveur Flask

Un système de logs enregistre les erreurs et crashs dans logs/atlas.log.
"""

import hashlib
import json
import logging
import os
import sys
import traceback

# Force stdout line-buffering so log lines are never lost when the process crashes
# (Pterodactyl captures stdout via a pipe, which defaults to block-buffering)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
else:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

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

ATLAS_VERSION = "2.0.0"
CHEMIN_MODELE = os.path.join("brain", "model.json")
CHEMIN_INTENTS = os.path.join("data", "intents.json")
CHEMIN_HASH = os.path.join("brain", "intents.hash")


# ─────────────────────────────────────────────────
# Bannière ASCII
# ─────────────────────────────────────────────────


def afficher_banniere():
    """Affiche la bannière ASCII d'Atlas au démarrage."""
    banniere = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ██████╗████████╗██╗      █████╗ ███████╗                ║
║    ██╔══██╗╚══██╔══╝██║     ██╔══██╗██╔════╝                ║
║    ███████║   ██║   ██║     ███████║███████╗                 ║
║    ██╔══██║   ██║   ██║     ██╔══██║╚════██║                 ║
║    ██║  ██║   ██║   ███████╗██║  ██║███████║                 ║
║    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝                ║
║                                                              ║
║          Intelligence Artificielle — v{ATLAS_VERSION}              ║
║          Réseau neuronal 100% from scratch (NumPy)           ║
║          Optimiseur : Adam | Architecture : 4 couches        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝"""
    print(banniere)


# ─────────────────────────────────────────────────
# Détection robuste de l'IP et du port
# ─────────────────────────────────────────────────


def detecter_host_port():
    """
    Détecte l'adresse de bind et le port de manière robuste.
    Pterodactyl définit SERVER_IP (IP externe) et SERVER_PORT.
    Le serveur bind TOUJOURS sur 0.0.0.0 à l'intérieur du conteneur ;
    SERVER_IP est utilisé uniquement pour l'affichage dans les logs.
    Ne crash jamais : retourne les valeurs par défaut en cas de problème.
    """
    bind_host = os.environ.get("HOST", "0.0.0.0")
    display_host = os.environ.get("SERVER_IP") or bind_host

    port_str = os.environ.get("SERVER_PORT") or os.environ.get("PORT", "7778")
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f"Port hors limites : {port}")
    except (ValueError, TypeError) as e:
        logger.warning("Port invalide '%s' (%s). Utilisation du port par défaut 7778.", port_str, e)
        port = 7778

    return bind_host, port, display_host


# ─────────────────────────────────────────────────
# Gestion du hash des intents (détection de changement)
# ─────────────────────────────────────────────────


def _hash_intents():
    """Calcule le hash MD5 du fichier intents.json."""
    try:
        with open(CHEMIN_INTENTS, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def modele_est_a_jour():
    """
    Vérifie si le modèle sauvegardé correspond aux intents actuels.
    Retourne False si le modèle est absent, vide, ou si les intents ont changé.
    """
    if not os.path.isfile(CHEMIN_MODELE):
        return False
    try:
        taille = os.path.getsize(CHEMIN_MODELE)
        if taille == 0:
            return False
    except OSError:
        return False

    hash_actuel = _hash_intents()
    if hash_actuel is None:
        return True  # pas d'intents = pas besoin de ré-entraîner

    if os.path.isfile(CHEMIN_HASH):
        try:
            with open(CHEMIN_HASH, "r") as f:
                hash_sauvegarde = f.read().strip()
            return hash_sauvegarde == hash_actuel
        except OSError:
            pass
    return False  # pas de hash = modèle potentiellement périmé


def sauvegarder_hash_intents():
    """Sauvegarde le hash courant du fichier intents.json."""
    hash_actuel = _hash_intents()
    if hash_actuel:
        os.makedirs(os.path.dirname(CHEMIN_HASH) or ".", exist_ok=True)
        with open(CHEMIN_HASH, "w") as f:
            f.write(hash_actuel)


# ─────────────────────────────────────────────────
# Entraînement du modèle
# ─────────────────────────────────────────────────


def entrainer_modele():
    """Lance l'entraînement si le modèle n'est pas à jour. Retourne True si OK."""
    if modele_est_a_jour():
        try:
            taille = os.path.getsize(CHEMIN_MODELE)
            logger.info("✅ Modèle à jour : %s (%d octets)", CHEMIN_MODELE, taille)
            return True
        except OSError as e:
            logger.warning("Impossible de lire le modèle : %s", e)

    logger.info("🔄 Modèle absent ou périmé — lancement de l'entraînement...")

    # Compter les intents pour affichage
    try:
        with open(CHEMIN_INTENTS, "r", encoding="utf-8") as f:
            donnees = json.load(f)
        nb_intents = len(donnees.get("intents", []))
        logger.info("📚 Base de connaissances : %d intents", nb_intents)
    except Exception:
        nb_intents = "?"

    try:
        from brain.trainer import Trainer

        trainer = Trainer(
            chemin_intents=CHEMIN_INTENTS,
            chemin_modele=CHEMIN_MODELE,
            chemin_vocab="data/vocabulary.json",
            taille_cachee1=256,
            taille_cachee2=128,
            taille_cachee3=64,
            taux_apprentissage=0.001,
            taux_dropout=0.2,
            epochs=3000,
            taille_batch=32,
        )
        trainer.lancer()
        sauvegarder_hash_intents()
        logger.info("✅ Entraînement terminé avec succès !")
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

    host, port, display_host = detecter_host_port()
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info("🌐 Atlas démarre sur %s:%d (bind: %s) ...", display_host, port, host)

    try:
        app.run(host=host, port=port, debug=debug)
    except OSError as e:
        logger.error("Impossible de démarrer le serveur sur %s:%d — %s", display_host, port, e)
        return False
    except Exception:
        logger.error("Crash du serveur Flask :\n%s", traceback.format_exc())
        return False

    return True


# ─────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────


def main():
    afficher_banniere()
    logger.info("Atlas v%s — Démarrage", ATLAS_VERSION)
    logger.info("==============================================")

    # Étape 1 : entraînement si nécessaire
    if not entrainer_modele():
        logger.error("❌ Arrêt : l'entraînement a échoué.")
        return 1

    logger.info("🚀 Lancement du serveur Flask...")
    logger.info("==============================================")

    # Étape 2 : lancement du serveur
    if not lancer_serveur():
        logger.error("❌ Arrêt : le serveur n'a pas pu démarrer.")
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
