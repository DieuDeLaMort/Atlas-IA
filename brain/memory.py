"""
Système de mémoire d'Atlas — inspiré de Jarvis.
Gère la mémoire à court terme (conversation) et à long terme (faits appris, préférences).
Les données sont persistées automatiquement dans data/memory/ et peuvent être
synchronisées avec le dépôt Git.

Architecture mémoire :
  - Mémoire court terme  : contexte de conversation récent (derniers échanges)
  - Mémoire long terme   : faits appris, préférences utilisateur, historique
  - Auto-sauvegarde       : chaque modification est persistée en JSON
"""

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime

logger = logging.getLogger("atlas.memory")

# ─────────────────────────────────────────────────
# Chemins par défaut
# ─────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "data", "memory")
COURT_TERME_PATH = os.path.join(MEMORY_DIR, "court_terme.json")
LONG_TERME_PATH = os.path.join(MEMORY_DIR, "long_terme.json")
PREFERENCES_PATH = os.path.join(MEMORY_DIR, "preferences.json")
HISTORIQUE_PATH = os.path.join(MEMORY_DIR, "historique.json")
FAITS_APPRIS_PATH = os.path.join(MEMORY_DIR, "faits_appris.json")


class MemoireCourtTerme:
    """
    Mémoire à court terme — garde le contexte conversationnel récent.
    Fonctionne comme un buffer circulaire de taille fixe.
    """

    def __init__(self, capacite=50):
        """
        :param capacite: Nombre maximum d'échanges conservés en mémoire court terme.
        """
        self.capacite = capacite
        self.echanges = []
        self.contexte_actuel = None
        self.derniere_intention = None
        self.sujet_en_cours = None
        self._lock = threading.Lock()

    def ajouter_echange(self, message_utilisateur, reponse_atlas, intention=None, confiance=0.0):
        """
        Ajoute un échange à la mémoire court terme.

        :param message_utilisateur: Message de l'utilisateur
        :param reponse_atlas:       Réponse d'Atlas
        :param intention:           Tag d'intention détecté
        :param confiance:           Score de confiance
        """
        with self._lock:
            echange = {
                "timestamp": datetime.now().isoformat(),
                "utilisateur": message_utilisateur,
                "atlas": reponse_atlas,
                "intention": intention,
                "confiance": confiance,
            }
            self.echanges.append(echange)

            # Buffer circulaire
            if len(self.echanges) > self.capacite:
                self.echanges = self.echanges[-self.capacite:]

            self.derniere_intention = intention
            if intention:
                self.sujet_en_cours = intention

    def obtenir_contexte(self, n=5):
        """
        Retourne les n derniers échanges pour le contexte conversationnel.

        :param n: Nombre d'échanges à retourner
        :return:  Liste des derniers échanges
        """
        with self._lock:
            return self.echanges[-n:]

    def obtenir_dernier_sujet(self):
        """Retourne le dernier sujet abordé."""
        return self.sujet_en_cours

    def reinitialiser(self):
        """Efface la mémoire court terme."""
        with self._lock:
            self.echanges.clear()
            self.contexte_actuel = None
            self.derniere_intention = None
            self.sujet_en_cours = None

    def to_dict(self):
        """Sérialise la mémoire court terme."""
        with self._lock:
            return {
                "capacite": self.capacite,
                "echanges": self.echanges,
                "contexte_actuel": self.contexte_actuel,
                "derniere_intention": self.derniere_intention,
                "sujet_en_cours": self.sujet_en_cours,
            }


class MemoireLongTerme:
    """
    Mémoire à long terme — stocke les faits appris, préférences utilisateur,
    et l'historique des interactions. Persistée automatiquement sur disque.
    """

    def __init__(self):
        self.faits = {}           # {sujet: [liste de faits]}
        self.preferences = {}     # {cle: valeur}
        self.historique = []      # Liste chronologique d'événements
        self.compteur_interactions = 0
        self.date_premiere_interaction = None
        self.sujets_frequents = {}  # {sujet: compteur}
        self._lock = threading.Lock()
        self._charger()

    def _charger(self):
        """Charge la mémoire long terme depuis les fichiers JSON."""
        try:
            if os.path.isfile(FAITS_APPRIS_PATH):
                with open(FAITS_APPRIS_PATH, "r", encoding="utf-8") as f:
                    self.faits = json.load(f)
                logger.info("Mémoire : %d sujets chargés depuis faits_appris.json", len(self.faits))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Impossible de charger faits_appris.json : %s", e)

        try:
            if os.path.isfile(PREFERENCES_PATH):
                with open(PREFERENCES_PATH, "r", encoding="utf-8") as f:
                    self.preferences = json.load(f)
                logger.info("Mémoire : %d préférences chargées", len(self.preferences))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Impossible de charger preferences.json : %s", e)

        try:
            if os.path.isfile(HISTORIQUE_PATH):
                with open(HISTORIQUE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.historique = data.get("evenements", [])
                self.compteur_interactions = data.get("compteur_interactions", 0)
                self.date_premiere_interaction = data.get("date_premiere_interaction")
                self.sujets_frequents = data.get("sujets_frequents", {})
                logger.info("Mémoire : historique chargé (%d événements)", len(self.historique))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Impossible de charger historique.json : %s", e)

    def sauvegarder(self):
        """Persiste toute la mémoire long terme sur disque."""
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with self._lock:
            try:
                with open(FAITS_APPRIS_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.faits, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.error("Erreur sauvegarde faits : %s", e)

            try:
                with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.error("Erreur sauvegarde préférences : %s", e)

            try:
                historique_data = {
                    "compteur_interactions": self.compteur_interactions,
                    "date_premiere_interaction": self.date_premiere_interaction,
                    "sujets_frequents": self.sujets_frequents,
                    "evenements": self.historique[-500:],  # Garder les 500 derniers
                }
                with open(HISTORIQUE_PATH, "w", encoding="utf-8") as f:
                    json.dump(historique_data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.error("Erreur sauvegarde historique : %s", e)

        logger.info("Mémoire long terme sauvegardée.")

    def apprendre_fait(self, sujet, fait):
        """
        Enregistre un nouveau fait dans la mémoire long terme.

        :param sujet: Catégorie du fait (ex: "utilisateur", "science", "preferences")
        :param fait:  Le fait à mémoriser
        """
        with self._lock:
            if sujet not in self.faits:
                self.faits[sujet] = []
            # Éviter les doublons
            if fait not in self.faits[sujet]:
                self.faits[sujet].append(fait)
                logger.info("Fait appris — [%s] : %s", sujet, fait[:80])
        self.sauvegarder()

    def obtenir_faits(self, sujet=None):
        """
        Récupère les faits mémorisés.

        :param sujet: Si spécifié, retourne uniquement les faits de ce sujet
        :return:      Dictionnaire de faits ou liste
        """
        with self._lock:
            if sujet:
                return self.faits.get(sujet, [])
            return dict(self.faits)

    def definir_preference(self, cle, valeur):
        """
        Enregistre une préférence utilisateur.

        :param cle:    Clé de la préférence
        :param valeur: Valeur de la préférence
        """
        with self._lock:
            self.preferences[cle] = valeur
            logger.info("Préférence définie — %s : %s", cle, valeur)
        self.sauvegarder()

    def obtenir_preference(self, cle, defaut=None):
        """Récupère une préférence utilisateur."""
        with self._lock:
            return self.preferences.get(cle, defaut)

    def enregistrer_interaction(self, intention, message):
        """
        Enregistre une interaction dans l'historique et met à jour les stats.

        :param intention: Tag d'intention
        :param message:   Message de l'utilisateur
        """
        with self._lock:
            self.compteur_interactions += 1
            if not self.date_premiere_interaction:
                self.date_premiere_interaction = datetime.now().isoformat()

            if intention:
                self.sujets_frequents[intention] = self.sujets_frequents.get(intention, 0) + 1

            evenement = {
                "timestamp": datetime.now().isoformat(),
                "intention": intention,
                "message_court": message[:100] if message else "",
            }
            self.historique.append(evenement)

            # Auto-sauvegarde toutes les 10 interactions
            if self.compteur_interactions % 10 == 0:
                self._sauvegarder_sans_lock()

    def _sauvegarder_sans_lock(self):
        """Sauvegarde interne (appelée quand le lock est déjà acquis)."""
        os.makedirs(MEMORY_DIR, exist_ok=True)
        try:
            with open(FAITS_APPRIS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.faits, f, ensure_ascii=False, indent=2)
            with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            historique_data = {
                "compteur_interactions": self.compteur_interactions,
                "date_premiere_interaction": self.date_premiere_interaction,
                "sujets_frequents": self.sujets_frequents,
                "evenements": self.historique[-500:],
            }
            with open(HISTORIQUE_PATH, "w", encoding="utf-8") as f:
                json.dump(historique_data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("Erreur auto-sauvegarde : %s", e)

    def obtenir_statistiques(self):
        """Retourne les statistiques d'interaction."""
        with self._lock:
            top_sujets = sorted(
                self.sujets_frequents.items(), key=lambda x: x[1], reverse=True
            )[:10]
            return {
                "total_interactions": self.compteur_interactions,
                "premiere_interaction": self.date_premiere_interaction,
                "sujets_frequents": dict(top_sujets),
                "faits_memorises": sum(len(v) for v in self.faits.values()),
                "preferences_definies": len(self.preferences),
            }


class GestionnaireMemoire:
    """
    Gestionnaire principal de mémoire — orchestre court terme et long terme.
    Point d'entrée unique pour toutes les opérations mémoire.
    """

    def __init__(self, capacite_court_terme=50):
        self.court_terme = MemoireCourtTerme(capacite=capacite_court_terme)
        self.long_terme = MemoireLongTerme()
        self._auto_save_thread = None
        self._running = False

    def demarrer_auto_sauvegarde(self, intervalle=300):
        """
        Démarre un thread de sauvegarde automatique.

        :param intervalle: Intervalle en secondes entre chaque sauvegarde (défaut: 5 min)
        """
        if self._running:
            return

        self._running = True

        def _boucle():
            while self._running:
                time.sleep(intervalle)
                if self._running:
                    self.sauvegarder_tout()
                    self._git_auto_commit()

        self._auto_save_thread = threading.Thread(target=_boucle, daemon=True)
        self._auto_save_thread.start()
        logger.info("Auto-sauvegarde mémoire activée (intervalle: %ds)", intervalle)

    def arreter_auto_sauvegarde(self):
        """Arrête le thread de sauvegarde automatique."""
        self._running = False
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5)

    def enregistrer_echange(self, message, reponse, intention=None, confiance=0.0):
        """
        Enregistre un échange complet (court terme + long terme).

        :param message:   Message utilisateur
        :param reponse:   Réponse d'Atlas
        :param intention: Tag d'intention
        :param confiance: Score de confiance
        """
        self.court_terme.ajouter_echange(message, reponse, intention, confiance)
        self.long_terme.enregistrer_interaction(intention, message)

    def apprendre(self, sujet, fait):
        """Délègue l'apprentissage à la mémoire long terme."""
        self.long_terme.apprendre_fait(sujet, fait)

    def se_souvenir(self, sujet=None):
        """Récupère les faits mémorisés."""
        return self.long_terme.obtenir_faits(sujet)

    def definir_preference(self, cle, valeur):
        """Enregistre une préférence."""
        self.long_terme.definir_preference(cle, valeur)

    def obtenir_preference(self, cle, defaut=None):
        """Récupère une préférence."""
        return self.long_terme.obtenir_preference(cle, defaut)

    def obtenir_contexte_conversation(self, n=5):
        """Retourne les n derniers échanges."""
        return self.court_terme.obtenir_contexte(n)

    def obtenir_statistiques(self):
        """Retourne les statistiques complètes."""
        stats = self.long_terme.obtenir_statistiques()
        stats["echanges_en_memoire_court_terme"] = len(self.court_terme.echanges)
        stats["sujet_en_cours"] = self.court_terme.obtenir_dernier_sujet()
        return stats

    def sauvegarder_tout(self):
        """Force la sauvegarde complète."""
        os.makedirs(MEMORY_DIR, exist_ok=True)

        # Sauvegarder mémoire court terme
        try:
            with open(COURT_TERME_PATH, "w", encoding="utf-8") as f:
                json.dump(self.court_terme.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("Erreur sauvegarde court terme : %s", e)

        # Sauvegarder mémoire long terme
        self.long_terme.sauvegarder()
        logger.info("Sauvegarde complète de la mémoire effectuée.")

    def _git_auto_commit(self):
        """
        Tente un auto-commit Git des fichiers mémoire.
        Silencieux en cas d'échec (pas de Git, pas de droits, etc.)
        """
        try:
            memory_files = [
                COURT_TERME_PATH,
                LONG_TERME_PATH,
                PREFERENCES_PATH,
                HISTORIQUE_PATH,
                FAITS_APPRIS_PATH,
            ]
            existing = [f for f in memory_files if os.path.isfile(f)]
            if not existing:
                return

            subprocess.run(
                ["git", "add"] + existing,
                cwd=BASE_DIR,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", "Atlas: auto-sauvegarde mémoire"],
                cwd=BASE_DIR,
                capture_output=True,
                timeout=10,
            )
            logger.info("Mémoire auto-commitée dans le dépôt Git.")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("Auto-commit Git ignoré : %s", e)

    def exporter_memoire(self):
        """Exporte toute la mémoire sous forme de dictionnaire."""
        return {
            "court_terme": self.court_terme.to_dict(),
            "long_terme": {
                "faits": self.long_terme.faits,
                "preferences": self.long_terme.preferences,
                "statistiques": self.long_terme.obtenir_statistiques(),
            },
        }

    def rechercher_dans_memoire(self, mot_cle):
        """
        Recherche un mot-clé dans toute la mémoire.

        :param mot_cle: Mot-clé à rechercher
        :return:        Liste de résultats trouvés
        """
        resultats = []
        mot_cle_lower = mot_cle.lower()

        # Chercher dans les faits
        for sujet, faits in self.long_terme.faits.items():
            for fait in faits:
                if mot_cle_lower in fait.lower():
                    resultats.append({"source": "fait", "sujet": sujet, "contenu": fait})

        # Chercher dans les échanges récents
        for echange in self.court_terme.echanges:
            if mot_cle_lower in echange.get("utilisateur", "").lower():
                resultats.append({
                    "source": "echange",
                    "contenu": echange["utilisateur"],
                    "timestamp": echange.get("timestamp"),
                })

        return resultats
