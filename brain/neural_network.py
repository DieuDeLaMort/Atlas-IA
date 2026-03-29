"""
Réseau de neurones multi-couches implémenté from scratch avec NumPy uniquement.
Architecture : input → hidden1 → hidden2 → output
Activation    : ReLU pour les couches cachées, Softmax pour la sortie
"""

import json
import numpy as np


class ReseauNeuronal:
    """
    Réseau de neurones entièrement codé from scratch.
    Pas de TensorFlow, pas de PyTorch — uniquement NumPy.
    """

    def __init__(self, taille_entree, taille_cachee1, taille_cachee2, taille_sortie, taux_apprentissage=0.01):
        """
        Initialise le réseau avec des poids aléatoires (He initialization).

        :param taille_entree:   Nombre de neurones d'entrée (taille du vecteur bag-of-words)
        :param taille_cachee1:  Nombre de neurones dans la première couche cachée
        :param taille_cachee2:  Nombre de neurones dans la deuxième couche cachée
        :param taille_sortie:   Nombre de neurones de sortie (nombre d'intents)
        :param taux_apprentissage: Learning rate pour la descente de gradient
        """
        self.taille_entree = taille_entree
        self.taille_cachee1 = taille_cachee1
        self.taille_cachee2 = taille_cachee2
        self.taille_sortie = taille_sortie
        self.taux_apprentissage = taux_apprentissage

        # Initialisation des poids avec He initialization (adapté pour ReLU)
        self.poids1 = np.random.randn(taille_entree, taille_cachee1) * np.sqrt(2.0 / taille_entree)
        self.biais1 = np.zeros((1, taille_cachee1))

        self.poids2 = np.random.randn(taille_cachee1, taille_cachee2) * np.sqrt(2.0 / taille_cachee1)
        self.biais2 = np.zeros((1, taille_cachee2))

        self.poids3 = np.random.randn(taille_cachee2, taille_sortie) * np.sqrt(2.0 / taille_cachee2)
        self.biais3 = np.zeros((1, taille_sortie))

        # Stockage des activations pour la rétropropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

    # ─────────────────────────────────────────────────
    # Fonctions d'activation
    # ─────────────────────────────────────────────────

    def relu(self, x):
        """Fonction d'activation ReLU : max(0, x)"""
        return np.maximum(0, x)

    def relu_derivee(self, x):
        """Dérivée de ReLU : 1 si x > 0, sinon 0"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """
        Fonction Softmax stable numériquement.
        Transforme les scores bruts en probabilités.
        """
        # Soustraction du maximum pour la stabilité numérique
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # ─────────────────────────────────────────────────
    # Propagation avant (Forward Pass)
    # ─────────────────────────────────────────────────

    def forward(self, X):
        """
        Calcule la sortie du réseau pour une entrée X.

        :param X: Matrice d'entrée de forme (n_echantillons, taille_entree)
        :return:  Probabilités de forme (n_echantillons, taille_sortie)
        """
        # Couche 1 : input → hidden1
        self.z1 = np.dot(X, self.poids1) + self.biais1
        self.a1 = self.relu(self.z1)

        # Couche 2 : hidden1 → hidden2
        self.z2 = np.dot(self.a1, self.poids2) + self.biais2
        self.a2 = self.relu(self.z2)

        # Couche 3 : hidden2 → output
        self.z3 = np.dot(self.a2, self.poids3) + self.biais3
        self.a3 = self.softmax(self.z3)

        return self.a3

    # ─────────────────────────────────────────────────
    # Fonction de perte (Cross-Entropy Loss)
    # ─────────────────────────────────────────────────

    def calculer_perte(self, y_pred, y_reel):
        """
        Calcule la perte cross-entropique.

        :param y_pred: Probabilités prédites (n_echantillons, n_classes)
        :param y_reel: Labels réels encodés one-hot (n_echantillons, n_classes)
        :return:       Valeur scalaire de la perte
        """
        n = y_reel.shape[0]
        # Clip pour éviter log(0)
        y_pred_clip = np.clip(y_pred, 1e-12, 1.0)
        perte = -np.sum(y_reel * np.log(y_pred_clip)) / n
        return perte

    # ─────────────────────────────────────────────────
    # Rétropropagation (Backward Pass)
    # ─────────────────────────────────────────────────

    def backward(self, X, y_reel):
        """
        Calcule les gradients et met à jour les poids via descente de gradient.

        :param X:      Matrice d'entrée (n_echantillons, taille_entree)
        :param y_reel: Labels one-hot (n_echantillons, n_classes)
        """
        n = X.shape[0]

        # ── Gradient de la couche de sortie (Softmax + Cross-Entropy) ──
        # La dérivée combinée Softmax + CrossEntropy est simplement : a3 - y
        dz3 = (self.a3 - y_reel) / n

        dpoids3 = np.dot(self.a2.T, dz3)
        dbiais3 = np.sum(dz3, axis=0, keepdims=True)

        # ── Gradient couche cachée 2 ──
        da2 = np.dot(dz3, self.poids3.T)
        dz2 = da2 * self.relu_derivee(self.z2)

        dpoids2 = np.dot(self.a1.T, dz2)
        dbiais2 = np.sum(dz2, axis=0, keepdims=True)

        # ── Gradient couche cachée 1 ──
        da1 = np.dot(dz2, self.poids2.T)
        dz1 = da1 * self.relu_derivee(self.z1)

        dpoids1 = np.dot(X.T, dz1)
        dbiais1 = np.sum(dz1, axis=0, keepdims=True)

        # ── Mise à jour des poids (descente de gradient) ──
        self.poids3 -= self.taux_apprentissage * dpoids3
        self.biais3 -= self.taux_apprentissage * dbiais3

        self.poids2 -= self.taux_apprentissage * dpoids2
        self.biais2 -= self.taux_apprentissage * dbiais2

        self.poids1 -= self.taux_apprentissage * dpoids1
        self.biais1 -= self.taux_apprentissage * dbiais1

    # ─────────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────────

    def entrainer(self, X, y, epochs=1000, afficher_progression=True):
        """
        Entraîne le réseau sur les données fournies.

        :param X:                  Données d'entrée (n_echantillons, taille_entree)
        :param y:                  Labels one-hot (n_echantillons, n_classes)
        :param epochs:             Nombre d'epochs d'entraînement
        :param afficher_progression: Affiche la perte/précision toutes les 100 epochs
        """
        for epoch in range(epochs):
            # Propagation avant
            y_pred = self.forward(X)

            # Calcul de la perte
            perte = self.calculer_perte(y_pred, y)

            # Rétropropagation
            self.backward(X, y)

            # Affichage de la progression
            if afficher_progression and (epoch + 1) % 100 == 0:
                predictions = np.argmax(y_pred, axis=1)
                labels_reels = np.argmax(y, axis=1)
                precision = np.mean(predictions == labels_reels) * 100
                print(f"Epoch {epoch + 1:5d}/{epochs} — Perte: {perte:.6f} — Précision: {precision:.1f}%")

    # ─────────────────────────────────────────────────
    # Prédiction
    # ─────────────────────────────────────────────────

    def predire(self, X):
        """
        Prédit la classe pour un vecteur d'entrée X.

        :param X: Vecteur d'entrée (1, taille_entree)
        :return:  (indice_classe, confiance) — indice de l'intent prédit et probabilité
        """
        probabilites = self.forward(X)
        indice = int(np.argmax(probabilites))
        confiance = float(probabilites[0][indice])
        return indice, confiance

    # ─────────────────────────────────────────────────
    # Sauvegarde / Chargement des poids
    # ─────────────────────────────────────────────────

    def sauvegarder(self, chemin="brain/model.json"):
        """
        Sauvegarde tous les poids et biais du réseau dans un fichier JSON.

        :param chemin: Chemin du fichier de sauvegarde
        """
        modele = {
            "architecture": {
                "taille_entree": self.taille_entree,
                "taille_cachee1": self.taille_cachee1,
                "taille_cachee2": self.taille_cachee2,
                "taille_sortie": self.taille_sortie,
                "taux_apprentissage": self.taux_apprentissage
            },
            "poids": {
                "poids1": self.poids1.tolist(),
                "biais1": self.biais1.tolist(),
                "poids2": self.poids2.tolist(),
                "biais2": self.biais2.tolist(),
                "poids3": self.poids3.tolist(),
                "biais3": self.biais3.tolist()
            }
        }
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(modele, f, ensure_ascii=False, indent=2)
        print(f"✅ Modèle sauvegardé dans '{chemin}'")

    def charger(self, chemin="brain/model.json"):
        """
        Charge les poids depuis un fichier JSON.

        :param chemin: Chemin du fichier de sauvegarde
        """
        with open(chemin, "r", encoding="utf-8") as f:
            modele = json.load(f)

        # Restaurer l'architecture
        arch = modele["architecture"]
        self.taille_entree = arch["taille_entree"]
        self.taille_cachee1 = arch["taille_cachee1"]
        self.taille_cachee2 = arch["taille_cachee2"]
        self.taille_sortie = arch["taille_sortie"]
        self.taux_apprentissage = arch["taux_apprentissage"]

        # Restaurer les poids
        poids = modele["poids"]
        self.poids1 = np.array(poids["poids1"])
        self.biais1 = np.array(poids["biais1"])
        self.poids2 = np.array(poids["poids2"])
        self.biais2 = np.array(poids["biais2"])
        self.poids3 = np.array(poids["poids3"])
        self.biais3 = np.array(poids["biais3"])

        print(f"✅ Modèle chargé depuis '{chemin}'")
