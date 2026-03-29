"""
Module d'entraînement d'Atlas.
Charge les données depuis data/intents.json,
prépare les vecteurs BoW et entraîne le réseau de neurones.
"""

import json
import os
import numpy as np

from brain.neural_network import ReseauNeuronal
from brain.tokenizer import Tokenizer


class Trainer:
    """
    Orchestre la préparation des données et l'entraînement du réseau neuronal d'Atlas.
    """

    def __init__(
        self,
        chemin_intents="data/intents.json",
        chemin_modele="brain/model.json",
        chemin_vocab="data/vocabulary.json",
        taille_cachee1=128,
        taille_cachee2=64,
        taux_apprentissage=0.01,
        epochs=5000,
    ):
        """
        :param chemin_intents:      Chemin vers la base de connaissances
        :param chemin_modele:       Chemin de sauvegarde du modèle
        :param chemin_vocab:        Chemin de sauvegarde du vocabulaire
        :param taille_cachee1:      Neurones de la 1ère couche cachée
        :param taille_cachee2:      Neurones de la 2ème couche cachée
        :param taux_apprentissage:  Learning rate
        :param epochs:              Nombre d'epochs d'entraînement
        """
        self.chemin_intents = chemin_intents
        self.chemin_modele = chemin_modele
        self.chemin_vocab = chemin_vocab
        self.taille_cachee1 = taille_cachee1
        self.taille_cachee2 = taille_cachee2
        self.taux_apprentissage = taux_apprentissage
        self.epochs = epochs

        self.tokenizer = Tokenizer()
        self.intents = []
        self.classes = []       # Liste des tags d'intent
        self.X_train = None     # Matrice de vecteurs BoW
        self.y_train = None     # Matrice one-hot des classes
        self.reseau = None

    # ─────────────────────────────────────────────────
    # Chargement des données
    # ─────────────────────────────────────────────────

    def charger_intents(self):
        """Charge le fichier intents.json et valide sa structure."""
        with open(self.chemin_intents, "r", encoding="utf-8") as f:
            donnees = json.load(f)

        self.intents = donnees.get("intents", [])
        if not self.intents:
            raise ValueError("Le fichier intents.json est vide ou mal formé.")

        self.classes = [intent["tag"] for intent in self.intents]
        print(f"📚 {len(self.intents)} intents chargés : {self.classes}")

    # ─────────────────────────────────────────────────
    # Préparation des données d'entraînement
    # ─────────────────────────────────────────────────

    def preparer_donnees(self):
        """
        Construit le vocabulaire et les matrices X_train / y_train.
        """
        # Collecter toutes les phrases pour construire le vocabulaire
        toutes_phrases = []
        for intent in self.intents:
            toutes_phrases.extend(intent.get("patterns", []))

        self.tokenizer.construire_vocabulaire(toutes_phrases)
        print(f"📖 Vocabulaire : {len(self.tokenizer.vocabulaire)} tokens uniques")

        # Construire les paires (vecteur BoW, label one-hot)
        X = []
        y = []

        for idx_intent, intent in enumerate(self.intents):
            for pattern in intent.get("patterns", []):
                vecteur = self.tokenizer.vectoriser(pattern)
                X.append(vecteur)

                # Label one-hot
                label = [0.0] * len(self.classes)
                label[idx_intent] = 1.0
                y.append(label)

        self.X_train = np.array(X)
        self.y_train = np.array(y)

        print(f"🔢 Données d'entraînement : {self.X_train.shape[0]} exemples, "
              f"{self.X_train.shape[1]} features → {self.y_train.shape[1]} classes")

    # ─────────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────────

    def entrainer(self):
        """Crée et entraîne le réseau de neurones."""
        taille_entree = self.X_train.shape[1]
        taille_sortie = self.y_train.shape[1]

        self.reseau = ReseauNeuronal(
            taille_entree=taille_entree,
            taille_cachee1=self.taille_cachee1,
            taille_cachee2=self.taille_cachee2,
            taille_sortie=taille_sortie,
            taux_apprentissage=self.taux_apprentissage,
        )

        print(f"\n🚀 Entraînement démarré ({self.epochs} epochs)...\n")
        self.reseau.entrainer(self.X_train, self.y_train, epochs=self.epochs)

    # ─────────────────────────────────────────────────
    # Sauvegarde
    # ─────────────────────────────────────────────────

    def sauvegarder(self):
        """Sauvegarde le modèle et le vocabulaire."""
        # Créer les dossiers si nécessaire
        os.makedirs(os.path.dirname(self.chemin_modele) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.chemin_vocab) or ".", exist_ok=True)

        self.reseau.sauvegarder(self.chemin_modele)
        self.tokenizer.sauvegarder(self.chemin_vocab)

        # Sauvegarder aussi les classes (ordre des intents)
        chemin_classes = os.path.join(os.path.dirname(self.chemin_modele), "classes.json")
        with open(chemin_classes, "w", encoding="utf-8") as f:
            json.dump(self.classes, f, ensure_ascii=False, indent=2)
        print(f"✅ Classes sauvegardées dans '{chemin_classes}'")

    # ─────────────────────────────────────────────────
    # Pipeline complet
    # ─────────────────────────────────────────────────

    def lancer(self):
        """Exécute le pipeline complet : chargement → préparation → entraînement → sauvegarde."""
        print("=" * 60)
        print("         🤖 ATLAS — Entraînement du cerveau")
        print("=" * 60)
        self.charger_intents()
        self.preparer_donnees()
        self.entrainer()
        self.sauvegarder()
        print("\n" + "=" * 60)
        print("  ✅ Entraînement terminé ! Atlas est prêt.")
        print("=" * 60)
