"""
Moteur d'apprentissage d'Atlas — capacité d'apprendre de nouvelles connaissances.
Permet à Atlas de :
  - Apprendre de nouveaux patterns et réponses en temps réel
  - Renforcer les intents existants avec de nouveaux exemples
  - Créer dynamiquement de nouvelles catégories d'intents
  - Sauvegarder les apprentissages dans le fichier intents et la mémoire
"""

import json
import logging
import os
import re
import threading
from datetime import datetime

logger = logging.getLogger("atlas.learning")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")
LEARNED_PATH = os.path.join(BASE_DIR, "data", "memory", "apprentissages.json")


class MoteurApprentissage:
    """
    Moteur d'apprentissage dynamique pour Atlas.
    Gère l'acquisition de nouvelles connaissances en temps réel.
    """

    def __init__(self):
        self.apprentissages = []          # Nouveaux patterns appris
        self.corrections = []             # Corrections faites par l'utilisateur
        self.nouveaux_intents = []        # Intents créés dynamiquement
        self.patterns_renforces = {}      # {tag: [nouveaux patterns]}
        self.reponses_ajoutees = {}       # {tag: [nouvelles réponses]}
        self._lock = threading.Lock()
        self._charger_apprentissages()

    def _charger_apprentissages(self):
        """Charge les apprentissages précédents depuis le fichier."""
        try:
            if os.path.isfile(LEARNED_PATH):
                with open(LEARNED_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.apprentissages = data.get("apprentissages", [])
                self.corrections = data.get("corrections", [])
                self.nouveaux_intents = data.get("nouveaux_intents", [])
                self.patterns_renforces = data.get("patterns_renforces", {})
                self.reponses_ajoutees = data.get("reponses_ajoutees", {})
                logger.info(
                    "Apprentissages chargés : %d patterns, %d corrections, %d nouveaux intents",
                    len(self.apprentissages),
                    len(self.corrections),
                    len(self.nouveaux_intents),
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Impossible de charger apprentissages.json : %s", e)

    def sauvegarder(self):
        """Persiste les apprentissages sur disque."""
        os.makedirs(os.path.dirname(LEARNED_PATH), exist_ok=True)
        with self._lock:
            data = {
                "derniere_mise_a_jour": datetime.now().isoformat(),
                "apprentissages": self.apprentissages,
                "corrections": self.corrections,
                "nouveaux_intents": self.nouveaux_intents,
                "patterns_renforces": self.patterns_renforces,
                "reponses_ajoutees": self.reponses_ajoutees,
            }
            try:
                with open(LEARNED_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info("Apprentissages sauvegardés.")
            except OSError as e:
                logger.error("Erreur sauvegarde apprentissages : %s", e)

    # ─────────────────────────────────────────────────
    # Apprentissage de nouvelles associations
    # ─────────────────────────────────────────────────

    def apprendre_association(self, question, reponse, tag=None):
        """
        Apprend une nouvelle association question/réponse.

        :param question: La question ou pattern d'entrée
        :param reponse:  La réponse correspondante
        :param tag:      Tag d'intent (optionnel, sera généré si absent)
        :return:         Dictionnaire décrivant l'apprentissage
        """
        with self._lock:
            if not tag:
                tag = self._generer_tag(question)

            apprentissage = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "reponse": reponse,
                "tag": tag,
            }
            self.apprentissages.append(apprentissage)

            # Ajouter aux patterns/réponses renforcés
            if tag not in self.patterns_renforces:
                self.patterns_renforces[tag] = []
            if question not in self.patterns_renforces[tag]:
                self.patterns_renforces[tag].append(question)

            if tag not in self.reponses_ajoutees:
                self.reponses_ajoutees[tag] = []
            if reponse not in self.reponses_ajoutees[tag]:
                self.reponses_ajoutees[tag].append(reponse)

        self.sauvegarder()
        logger.info("Nouvelle association apprise : [%s] %s → %s", tag, question[:50], reponse[:50])
        return apprentissage

    def enregistrer_correction(self, message_original, mauvaise_reponse, bonne_reponse, tag_suggere=None):
        """
        Enregistre une correction faite par l'utilisateur.

        :param message_original:  Le message qui a reçu une mauvaise réponse
        :param mauvaise_reponse:  La réponse incorrecte
        :param bonne_reponse:     La bonne réponse
        :param tag_suggere:       Tag d'intent suggéré
        """
        with self._lock:
            correction = {
                "timestamp": datetime.now().isoformat(),
                "message": message_original,
                "mauvaise_reponse": mauvaise_reponse,
                "bonne_reponse": bonne_reponse,
                "tag_suggere": tag_suggere,
            }
            self.corrections.append(correction)

        # Apprendre la bonne association
        if tag_suggere:
            self.apprendre_association(message_original, bonne_reponse, tag_suggere)
        else:
            self.sauvegarder()

        logger.info("Correction enregistrée pour : %s", message_original[:60])

    def creer_intent(self, tag, patterns, reponses):
        """
        Crée un nouvel intent dynamiquement.

        :param tag:      Nom unique de l'intent
        :param patterns: Liste de patterns d'entrée
        :param reponses: Liste de réponses possibles
        :return:         Le nouvel intent créé
        """
        with self._lock:
            nouvel_intent = {
                "tag": tag,
                "patterns": patterns,
                "responses": reponses,
                "learned": True,
                "created_at": datetime.now().isoformat(),
            }
            self.nouveaux_intents.append(nouvel_intent)

        self.sauvegarder()
        logger.info("Nouvel intent créé : %s (%d patterns, %d réponses)", tag, len(patterns), len(reponses))
        return nouvel_intent

    # ─────────────────────────────────────────────────
    # Intégration avec intents.json
    # ─────────────────────────────────────────────────

    def integrer_dans_intents(self):
        """
        Intègre les apprentissages dans le fichier intents.json.
        Renforce les intents existants et ajoute les nouveaux.
        """
        try:
            with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Impossible de lire intents.json : %s", e)
            return False

        intents = data.get("intents", [])
        tags_existants = {intent["tag"]: i for i, intent in enumerate(intents)}
        modifie = False

        # 1. Renforcer les intents existants avec de nouveaux patterns
        with self._lock:
            for tag, patterns in self.patterns_renforces.items():
                if tag in tags_existants:
                    idx = tags_existants[tag]
                    for pattern in patterns:
                        if pattern not in intents[idx]["patterns"]:
                            intents[idx]["patterns"].append(pattern)
                            modifie = True

            for tag, reponses in self.reponses_ajoutees.items():
                if tag in tags_existants:
                    idx = tags_existants[tag]
                    for reponse in reponses:
                        if reponse not in intents[idx]["responses"]:
                            intents[idx]["responses"].append(reponse)
                            modifie = True

            # 2. Ajouter les nouveaux intents
            for intent in self.nouveaux_intents:
                tag = intent["tag"]
                if tag not in tags_existants:
                    clean_intent = {
                        "tag": intent["tag"],
                        "patterns": intent["patterns"],
                        "responses": intent["responses"],
                    }
                    intents.append(clean_intent)
                    tags_existants[tag] = len(intents) - 1
                    modifie = True

        if modifie:
            data["intents"] = intents
            try:
                with open(INTENTS_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info("intents.json mis à jour avec les apprentissages.")
                return True
            except OSError as e:
                logger.error("Impossible d'écrire intents.json : %s", e)
                return False

        return True

    # ─────────────────────────────────────────────────
    # Détection de patterns d'apprentissage
    # ─────────────────────────────────────────────────

    def detecter_intention_apprentissage(self, message):
        """
        Détecte si l'utilisateur essaie d'enseigner quelque chose à Atlas.

        :param message: Message de l'utilisateur
        :return:        Dictionnaire avec les infos d'apprentissage ou None
        """
        message_lower = message.lower().strip()

        # Patterns de type "apprends que X c'est Y"
        patterns_apprendre = [
            r"(?:apprends?|retiens?|note|memorise|mémorise)\s+que\s+(.+?)(?:\s+(?:c'est|est|=|signifie|veut dire)\s+)(.+)",
            r"(?:quand je (?:dis?|demande))\s+[\"'](.+?)[\"']\s*(?:,?\s*(?:réponds?|dis))\s+[\"'](.+?)[\"']",
            r"(?:si (?:on|je|quelqu'un)\s+(?:dit|demande|écrit))\s+[\"'](.+?)[\"']\s*(?:,?\s*(?:tu (?:dois|peux) (?:répondre|dire)))\s+[\"'](.+?)[\"']",
        ]

        for pattern in patterns_apprendre:
            match = re.search(pattern, message_lower)
            if match:
                return {
                    "type": "association",
                    "question": match.group(1).strip(),
                    "reponse": match.group(2).strip(),
                }

        # Pattern "mon nom est X"
        nom_patterns = [
            r"(?:je m'appelle|mon (?:nom|prénom)\s+(?:c'est|est))\s+(\w+)",
            r"(?:appelle[- ]moi)\s+(\w+)",
        ]
        for pattern in nom_patterns:
            match = re.search(pattern, message_lower)
            if match:
                return {
                    "type": "preference",
                    "cle": "nom_utilisateur",
                    "valeur": match.group(1).strip().capitalize(),
                }

        # Pattern préférences
        pref_patterns = [
            r"(?:j'aime|j'adore|je préfère|ma? (?:couleur|jeu|film|série|musique|langage)\s+(?:préféré[e]?|favori(?:te)?)\s+(?:c'est|est))\s+(.+)",
        ]
        for pattern in pref_patterns:
            match = re.search(pattern, message_lower)
            if match:
                return {
                    "type": "preference",
                    "cle": "preference_generale",
                    "valeur": match.group(1).strip(),
                }

        return None

    # ─────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────

    def _generer_tag(self, texte):
        """Génère un tag d'intent à partir d'un texte."""
        # Nettoyage et simplification
        tag = texte.lower()
        tag = re.sub(r"[^a-z0-9\s]", "", tag)
        mots = tag.split()[:3]
        tag = "_".join(mots)
        if not tag:
            tag = f"appris_{len(self.apprentissages)}"
        return f"appris_{tag}"

    def obtenir_statistiques(self):
        """Retourne les statistiques d'apprentissage."""
        with self._lock:
            return {
                "associations_apprises": len(self.apprentissages),
                "corrections_enregistrees": len(self.corrections),
                "nouveaux_intents": len(self.nouveaux_intents),
                "patterns_renforces": sum(len(p) for p in self.patterns_renforces.values()),
                "reponses_ajoutees": sum(len(r) for r in self.reponses_ajoutees.values()),
            }
