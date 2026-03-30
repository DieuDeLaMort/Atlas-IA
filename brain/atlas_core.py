"""
Cœur d'Atlas — Intelligence artificielle autonome inspirée de Jarvis.
Orchestre tous les sous-systèmes : réseau neuronal, mémoire, apprentissage,
moteur de réponses, et recherche web.

Atlas est conçu pour :
  - Comprendre le langage naturel (français et anglais)
  - Répondre avec une personnalité Jarvis (professionnel, intelligent, légèrement sarcastique)
  - Apprendre de chaque interaction
  - Se souvenir du contexte conversationnel
  - Rechercher sur internet quand ses connaissances sont insuffisantes
  - Être activé par la voix avec le mot-clé "Atlas"
"""

import json
import logging
import os
import random
import traceback
from datetime import datetime

import numpy as np

from brain.neural_network import ReseauNeuronal
from brain.tokenizer import Tokenizer
from brain.memory import GestionnaireMemoire
from brain.learning import MoteurApprentissage
from brain.response_engine import MoteurReponses
from brain import web_search

logger = logging.getLogger("atlas.core")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_MODELE = os.path.join(BASE_DIR, "brain", "model.json")
CHEMIN_VOCAB = os.path.join(BASE_DIR, "data", "vocabulary.json")
CHEMIN_CLASSES = os.path.join(BASE_DIR, "brain", "classes.json")
CHEMIN_INTENTS = os.path.join(BASE_DIR, "data", "intents.json")

# Seuil de confiance pour le réseau neuronal
SEUIL_CONFIANCE = 0.35
SEUIL_CONFIANCE_ELEVE = 0.75


class AtlasCore:
    """
    Intelligence artificielle Atlas — le cerveau principal.
    Coordonne la compréhension, la mémoire, l'apprentissage et les réponses.
    """

    def __init__(self):
        """Initialise tous les sous-systèmes d'Atlas."""
        self.reseau = None
        self.tokenizer = None
        self.classes = []
        self.intents_map = {}
        self.intents_raw = []

        # Sous-systèmes
        self.memoire = GestionnaireMemoire(capacite_court_terme=100)
        self.apprentissage = MoteurApprentissage()
        self.moteur_reponses = MoteurReponses(memoire=self.memoire)

        # État
        self.est_pret = False
        self.date_demarrage = None
        self.compteur_requetes = 0

        logger.info("AtlasCore initialisé — chargement des sous-systèmes...")

    # ─────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────

    def demarrer(self):
        """
        Démarre Atlas — charge le modèle et active tous les systèmes.

        :return: True si le démarrage est réussi
        """
        logger.info("╔══════════════════════════════════════════════╗")
        logger.info("║      🤖 ATLAS — Démarrage du cœur IA       ║")
        logger.info("╚══════════════════════════════════════════════╝")

        # Charger le modèle neuronal
        if not self._charger_modele():
            logger.warning("Modèle neuronal non disponible. Atlas fonctionne en mode dégradé.")
        else:
            logger.info("✅ Réseau neuronal chargé.")

        # Activer la mémoire
        self.memoire.demarrer_auto_sauvegarde(intervalle=300)
        logger.info("✅ Système de mémoire activé.")

        # État
        self.est_pret = True
        self.date_demarrage = datetime.now()

        logger.info("✅ Atlas est en ligne et opérationnel.")
        logger.info("   — Réseau neuronal : %s", "✅" if self.reseau else "⚠️ Mode dégradé")
        logger.info("   — Classes d'intents : %d", len(self.classes))
        logger.info("   — Vocabulaire : %d tokens", len(self.tokenizer.vocabulaire) if self.tokenizer else 0)
        logger.info("   — Mémoire : Activée")
        logger.info("   — Apprentissage : Activé")
        logger.info("   — Moteur de réponses : Actif")

        return True

    def _charger_modele(self):
        """Charge le réseau neuronal, le vocabulaire et les intents."""
        fichiers = {
            "modele": CHEMIN_MODELE,
            "vocabulaire": CHEMIN_VOCAB,
            "classes": CHEMIN_CLASSES,
            "intents": CHEMIN_INTENTS,
        }

        for nom, chemin in fichiers.items():
            if not os.path.isfile(chemin):
                logger.warning("Fichier manquant : %s (%s)", nom, chemin)
                return False

        try:
            # Tokenizer
            self.tokenizer = Tokenizer()
            self.tokenizer.charger(CHEMIN_VOCAB)

            # Classes
            with open(CHEMIN_CLASSES, "r", encoding="utf-8") as f:
                self.classes = json.load(f)

            # Intents
            with open(CHEMIN_INTENTS, "r", encoding="utf-8") as f:
                donnees = json.load(f)
            self.intents_raw = donnees.get("intents", [])
            self.intents_map = {
                intent["tag"]: intent.get("responses", [])
                for intent in self.intents_raw
            }

            # Réseau neuronal
            self.reseau = ReseauNeuronal(1, 1, 1, 1)
            self.reseau.charger(CHEMIN_MODELE)

            return True
        except Exception:
            logger.error("Erreur chargement modèle :\n%s", traceback.format_exc())
            return False

    # ─────────────────────────────────────────────────
    # Traitement des messages
    # ─────────────────────────────────────────────────

    def traiter_message(self, message):
        """
        Point d'entrée principal — traite un message utilisateur et retourne une réponse.

        :param message: Message de l'utilisateur (texte)
        :return:        Dictionnaire {response, intention, confiance, source}
        """
        if not message or not message.strip():
            return self._construire_reponse(
                "Monsieur, je n'ai pas reçu de message. Pourriez-vous reformuler ?",
                intention=None,
                confiance=0.0,
                source="systeme",
            )

        self.compteur_requetes += 1
        message = message.strip()

        try:
            # 1. Vérifier si c'est une intention d'apprentissage
            apprentissage_info = self.apprentissage.detecter_intention_apprentissage(message)
            if apprentissage_info:
                return self._traiter_apprentissage(apprentissage_info, message)

            # 2. Vérifier les commandes spéciales
            commande = self._verifier_commande(message)
            if commande:
                return commande

            # 3. Prédiction par le réseau neuronal
            if self.reseau and self.tokenizer:
                intention, confiance = self._predire(message)

                if confiance >= SEUIL_CONFIANCE and intention in self.intents_map:
                    reponses = self.intents_map[intention]
                    contexte = self.memoire.obtenir_contexte_conversation(5)
                    reponse = self.moteur_reponses.generer_reponse_contextuelle(
                        reponses, intention, message, contexte
                    )
                    resultat = self._construire_reponse(reponse, intention, confiance, "reseau_neuronal")

                    # Enregistrer dans la mémoire
                    self.memoire.enregistrer_echange(message, reponse, intention, confiance)

                    return resultat

            # 4. Recherche web comme fallback
            logger.info("Recherche web pour : %s", message[:80])
            reponse_web = web_search.chercher(message)
            if reponse_web:
                reponse_finale = f"🌐 {reponse_web}"
                self.memoire.enregistrer_echange(message, reponse_finale, "recherche_web", 0.0)
                return self._construire_reponse(reponse_finale, "recherche_web", 0.0, "web")

            # 5. Aucune réponse trouvée
            reponse_defaut = self.moteur_reponses._reponse_par_defaut()
            self.memoire.enregistrer_echange(message, reponse_defaut, None, 0.0)
            return self._construire_reponse(reponse_defaut, None, 0.0, "defaut")

        except Exception:
            logger.error("Erreur traitement message :\n%s", traceback.format_exc())
            return self._construire_reponse(
                "⚠️ Une erreur interne est survenue. Mes excuses, Monsieur. Je me recalibre.",
                intention=None,
                confiance=0.0,
                source="erreur",
            )

    def _predire(self, message):
        """
        Utilise le réseau neuronal pour prédire l'intention.

        :param message: Message utilisateur
        :return:        (tag_intention, confiance)
        """
        vecteur = np.array([self.tokenizer.vectoriser(message)])
        indice, confiance = self.reseau.predire(vecteur)

        if 0 <= indice < len(self.classes):
            return self.classes[indice], confiance
        return None, 0.0

    def _construire_reponse(self, texte, intention, confiance, source):
        """Construit un dictionnaire de réponse standardisé."""
        return {
            "response": texte,
            "intention": intention,
            "confiance": round(confiance, 4) if confiance else 0.0,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────────────
    # Apprentissage en temps réel
    # ─────────────────────────────────────────────────

    def _traiter_apprentissage(self, info, message_original):
        """
        Traite une intention d'apprentissage détectée.

        :param info:             Dictionnaire d'info d'apprentissage
        :param message_original: Message brut
        :return:                 Réponse d'Atlas
        """
        type_apprentissage = info.get("type")

        if type_apprentissage == "association":
            question = info["question"]
            reponse = info["reponse"]
            self.apprentissage.apprendre_association(question, reponse)
            self.memoire.apprendre("associations", f"{question} → {reponse}")

            reponse_atlas = random.choice([
                f"Bien noté, Monsieur. J'ai appris que « {question} » correspond à « {reponse} ». "
                "Je m'en souviendrai.",
                f"Apprentissage enregistré. Quand on me dira « {question} », "
                f"je répondrai « {reponse} ». Merci de m'enseigner, Monsieur.",
                f"C'est noté dans ma mémoire. « {question} » → « {reponse} ». "
                "Mes capacités s'améliorent grâce à vous.",
            ])
            self.memoire.enregistrer_echange(message_original, reponse_atlas, "apprentissage", 1.0)
            return self._construire_reponse(reponse_atlas, "apprentissage", 1.0, "apprentissage")

        elif type_apprentissage == "preference":
            cle = info["cle"]
            valeur = info["valeur"]
            self.memoire.definir_preference(cle, valeur)

            if cle == "nom_utilisateur":
                reponse_atlas = random.choice([
                    f"Enchanté, {valeur}. Je me souviendrai de votre nom. "
                    "Puis-je vous être utile ?",
                    f"Bienvenue, {valeur}. Votre nom est désormais gravé dans ma mémoire. "
                    "Comment puis-je vous assister ?",
                    f"Noté. Je vous appellerai {valeur} désormais. À votre service.",
                ])
            else:
                reponse_atlas = f"Préférence enregistrée : {cle} = {valeur}. Bien noté, Monsieur."

            self.memoire.enregistrer_echange(message_original, reponse_atlas, "preference", 1.0)
            return self._construire_reponse(reponse_atlas, "preference", 1.0, "apprentissage")

        return self._construire_reponse(
            "J'ai détecté une intention d'apprentissage mais je n'ai pas compris le format. "
            "Reformulez, s'il vous plaît.",
            intention="apprentissage",
            confiance=0.5,
            source="apprentissage",
        )

    # ─────────────────────────────────────────────────
    # Commandes spéciales
    # ─────────────────────────────────────────────────

    def _verifier_commande(self, message):
        """
        Vérifie si le message est une commande spéciale.

        :param message: Message utilisateur
        :return:        Réponse si commande, None sinon
        """
        msg_lower = message.lower().strip()

        # Commande : statut système
        if msg_lower in (
            "statut", "status", "statut système", "statut systeme",
            "rapport", "diagnostique", "diagnostic", "état du système",
            "etat du systeme", "system status",
        ):
            rapport = self.moteur_reponses.generer_statut_systeme()
            return self._construire_reponse(rapport, "statut_systeme", 1.0, "systeme")

        # Commande : mémoire
        if msg_lower in (
            "que sais-tu sur moi", "ma mémoire", "ma memoire",
            "qu'as-tu retenu", "montre ta mémoire", "montre ta memoire",
        ):
            return self._afficher_memoire()

        # Commande : statistiques
        if msg_lower in (
            "statistiques", "stats", "tes stats", "tes statistiques",
            "combien d'interactions",
        ):
            return self._afficher_statistiques()

        # Commande : effacer mémoire court terme
        if msg_lower in (
            "oublie tout", "efface ta mémoire", "efface ta memoire",
            "reset mémoire", "reset memoire", "nouvelle conversation",
        ):
            self.memoire.court_terme.reinitialiser()
            reponse = "Mémoire court terme effacée. Nous repartons de zéro, Monsieur."
            return self._construire_reponse(reponse, "commande", 1.0, "systeme")

        return None

    def _afficher_memoire(self):
        """Affiche ce qu'Atlas sait sur l'utilisateur."""
        faits = self.memoire.se_souvenir()
        prefs = self.memoire.long_terme.preferences

        if not faits and not prefs:
            reponse = (
                "Ma mémoire vous concernant est vide pour le moment, Monsieur. "
                "Apprenez-moi des choses et je les retiendrai !"
            )
        else:
            lignes = ["📝 Voici ce que j'ai en mémoire :\n"]
            if prefs:
                lignes.append("**Préférences :**")
                for cle, val in prefs.items():
                    lignes.append(f"  • {cle} : {val}")
            if faits:
                lignes.append("\n**Faits appris :**")
                for sujet, liste_faits in faits.items():
                    lignes.append(f"  [{sujet}]")
                    for fait in liste_faits[-5:]:
                        lignes.append(f"    • {fait}")
            reponse = "\n".join(lignes)

        return self._construire_reponse(reponse, "memoire", 1.0, "systeme")

    def _afficher_statistiques(self):
        """Affiche les statistiques d'Atlas."""
        stats_memoire = self.memoire.obtenir_statistiques()
        stats_apprentissage = self.apprentissage.obtenir_statistiques()

        reponse = (
            f"📊 Statistiques Atlas :\n"
            f"• Requêtes traitées cette session : {self.compteur_requetes}\n"
            f"• Total interactions historiques : {stats_memoire.get('total_interactions', 0)}\n"
            f"• Faits mémorisés : {stats_memoire.get('faits_memorises', 0)}\n"
            f"• Préférences : {stats_memoire.get('preferences_definies', 0)}\n"
            f"• Associations apprises : {stats_apprentissage.get('associations_apprises', 0)}\n"
            f"• Corrections : {stats_apprentissage.get('corrections_enregistrees', 0)}\n"
            f"• Sujet en cours : {stats_memoire.get('sujet_en_cours', 'Aucun')}\n"
            f"• En ligne depuis : {self.date_demarrage.strftime('%H:%M') if self.date_demarrage else 'N/A'}"
        )
        return self._construire_reponse(reponse, "statistiques", 1.0, "systeme")

    # ─────────────────────────────────────────────────
    # API publique
    # ─────────────────────────────────────────────────

    def obtenir_salutation(self):
        """Génère une salutation de bienvenue."""
        return self.moteur_reponses.generer_salutation_jarvis()

    def est_operationnel(self):
        """Vérifie si Atlas est opérationnel."""
        return self.est_pret

    def arreter(self):
        """Arrête proprement Atlas."""
        logger.info("Arrêt d'Atlas en cours...")
        self.memoire.sauvegarder_tout()
        self.memoire.arreter_auto_sauvegarde()
        self.apprentissage.sauvegarder()
        logger.info("Atlas est arrêté. À bientôt, Monsieur.")
