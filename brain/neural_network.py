"""
Réseau de neurones multi-couches implémenté from scratch avec NumPy uniquement.
Architecture : input → hidden1 → hidden2 → hidden3 → output
Activation    : ReLU pour les couches cachées, Softmax pour la sortie
Optimiseur    : Adam (Adaptive Moment Estimation) — bien meilleur que le SGD simple
Régularisation: Dropout pendant l'entraînement pour éviter l'overfitting
"""

import json
import numpy as np


class ReseauNeuronal:
    """
    Réseau de neurones 4 couches codé from scratch.
    Architecture améliorée : 3 couches cachées + optimiseur Adam + Dropout.
    Pas de TensorFlow, pas de PyTorch — uniquement NumPy.
    """

    VERSION = "2.0"

    def __init__(
        self,
        taille_entree,
        taille_cachee1,
        taille_cachee2,
        taille_sortie,
        taille_cachee3=None,
        taux_apprentissage=0.001,
        taux_dropout=0.2,
    ):
        """
        Initialise le réseau avec des poids aléatoires (He initialization).

        :param taille_entree:    Nombre de neurones d'entrée (taille du vecteur bag-of-words)
        :param taille_cachee1:   Neurones dans la 1ère couche cachée
        :param taille_cachee2:   Neurones dans la 2ème couche cachée
        :param taille_sortie:    Nombre de neurones de sortie (nombre d'intents)
        :param taille_cachee3:   Neurones dans la 3ème couche cachée (None = désactivée)
        :param taux_apprentissage: Learning rate pour Adam
        :param taux_dropout:     Probabilité de désactivation des neurones (0 = pas de dropout)
        """
        self.taille_entree = taille_entree
        self.taille_cachee1 = taille_cachee1
        self.taille_cachee2 = taille_cachee2
        self.taille_cachee3 = taille_cachee3
        self.taille_sortie = taille_sortie
        self.taux_apprentissage = taux_apprentissage
        self.taux_dropout = taux_dropout
        self.trois_couches = taille_cachee3 is not None and taille_cachee3 > 0

        # ── Initialisation He pour ReLU ──
        self.poids1 = np.random.randn(taille_entree, taille_cachee1) * np.sqrt(2.0 / taille_entree)
        self.biais1 = np.zeros((1, taille_cachee1))

        self.poids2 = np.random.randn(taille_cachee1, taille_cachee2) * np.sqrt(2.0 / taille_cachee1)
        self.biais2 = np.zeros((1, taille_cachee2))

        if self.trois_couches:
            self.poids3 = np.random.randn(taille_cachee2, taille_cachee3) * np.sqrt(2.0 / taille_cachee2)
            self.biais3 = np.zeros((1, taille_cachee3))
            self.poids4 = np.random.randn(taille_cachee3, taille_sortie) * np.sqrt(2.0 / taille_cachee3)
            self.biais4 = np.zeros((1, taille_sortie))
        else:
            self.poids3 = np.random.randn(taille_cachee2, taille_sortie) * np.sqrt(2.0 / taille_cachee2)
            self.biais3 = np.zeros((1, taille_sortie))
            self.poids4 = None
            self.biais4 = None

        # ── Moments Adam (premier et second) pour chaque paramètre ──
        self._init_adam()

        # ── Stockage des activations pour la rétropropagation ──
        self.z1 = self.a1 = None
        self.z2 = self.a2 = None
        self.z3 = self.a3 = None
        self.z4 = self.a4 = None
        self.masque_dropout1 = self.masque_dropout2 = self.masque_dropout3 = None

        # Compteur d'étapes pour Adam (correction du biais)
        self._t = 0

    def _init_adam(self):
        """Initialise les moments Adam à zéro pour tous les poids."""
        self._m = {}
        self._v = {}
        for nom, p in self._params():
            self._m[nom] = np.zeros_like(p)
            self._v[nom] = np.zeros_like(p)

    def _params(self):
        """Itère sur tous les paramètres (nom, tableau) du réseau."""
        yield "poids1", self.poids1
        yield "biais1", self.biais1
        yield "poids2", self.poids2
        yield "biais2", self.biais2
        yield "poids3", self.poids3
        yield "biais3", self.biais3
        if self.trois_couches:
            yield "poids4", self.poids4
            yield "biais4", self.biais4

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
        """Softmax stable numériquement — transforme les scores en probabilités."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _dropout(self, a, taux, entrainement):
        """
        Applique le dropout inverted pendant l'entraînement.

        :param a:            Activation (n, d)
        :param taux:         Probabilité de mise à zéro
        :param entrainement: True pendant l'entraînement, False à l'inférence
        :return:             (activation_masquee, masque) — masque=None à l'inférence
        """
        if not entrainement or taux == 0.0:
            return a, None
        masque = (np.random.rand(*a.shape) > taux).astype(float)
        return a * masque / (1.0 - taux), masque

    # ─────────────────────────────────────────────────
    # Propagation avant (Forward Pass)
    # ─────────────────────────────────────────────────

    def forward(self, X, entrainement=False):
        """
        Calcule la sortie du réseau pour une entrée X.

        :param X:            Matrice d'entrée (n_echantillons, taille_entree)
        :param entrainement: Active le dropout si True
        :return:             Probabilités (n_echantillons, taille_sortie)
        """
        # Couche 1 : input → hidden1
        self.z1 = np.dot(X, self.poids1) + self.biais1
        a1_raw = self.relu(self.z1)
        self.a1, self.masque_dropout1 = self._dropout(a1_raw, self.taux_dropout, entrainement)

        # Couche 2 : hidden1 → hidden2
        self.z2 = np.dot(self.a1, self.poids2) + self.biais2
        a2_raw = self.relu(self.z2)
        self.a2, self.masque_dropout2 = self._dropout(a2_raw, self.taux_dropout, entrainement)

        if self.trois_couches:
            # Couche 3 : hidden2 → hidden3
            self.z3 = np.dot(self.a2, self.poids3) + self.biais3
            a3_raw = self.relu(self.z3)
            self.a3, self.masque_dropout3 = self._dropout(a3_raw, self.taux_dropout, entrainement)
            # Couche 4 : hidden3 → output
            self.z4 = np.dot(self.a3, self.poids4) + self.biais4
            self.a4 = self.softmax(self.z4)
            return self.a4
        else:
            # Couche 3 (sortie) : hidden2 → output
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
        y_pred_clip = np.clip(y_pred, 1e-12, 1.0)
        return -np.sum(y_reel * np.log(y_pred_clip)) / n

    # ─────────────────────────────────────────────────
    # Rétropropagation + mise à jour Adam
    # ─────────────────────────────────────────────────

    def backward(self, X, y_reel):
        """
        Calcule les gradients et met à jour les poids via Adam.

        :param X:      Matrice d'entrée (n_echantillons, taille_entree)
        :param y_reel: Labels one-hot (n_echantillons, n_classes)
        """
        n = X.shape[0]
        gradients = {}

        if self.trois_couches:
            # ── Sortie : hidden3 → output ──
            dz4 = (self.a4 - y_reel) / n
            gradients["poids4"] = np.dot(self.a3.T, dz4)
            gradients["biais4"] = np.sum(dz4, axis=0, keepdims=True)

            # ── Couche cachée 3 ──
            da3 = np.dot(dz4, self.poids4.T)
            if self.masque_dropout3 is not None:
                da3 = da3 * self.masque_dropout3 / (1.0 - self.taux_dropout)
            dz3 = da3 * self.relu_derivee(self.z3)
            gradients["poids3"] = np.dot(self.a2.T, dz3)
            gradients["biais3"] = np.sum(dz3, axis=0, keepdims=True)

            # ── Couche cachée 2 ──
            da2 = np.dot(dz3, self.poids3.T)
            if self.masque_dropout2 is not None:
                da2 = da2 * self.masque_dropout2 / (1.0 - self.taux_dropout)
            dz2 = da2 * self.relu_derivee(self.z2)
            gradients["poids2"] = np.dot(self.a1.T, dz2)
            gradients["biais2"] = np.sum(dz2, axis=0, keepdims=True)

            # ── Couche cachée 1 ──
            da1 = np.dot(dz2, self.poids2.T)
            if self.masque_dropout1 is not None:
                da1 = da1 * self.masque_dropout1 / (1.0 - self.taux_dropout)
            dz1 = da1 * self.relu_derivee(self.z1)
            gradients["poids1"] = np.dot(X.T, dz1)
            gradients["biais1"] = np.sum(dz1, axis=0, keepdims=True)
        else:
            # ── Sortie : hidden2 → output ──
            dz3 = (self.a3 - y_reel) / n
            gradients["poids3"] = np.dot(self.a2.T, dz3)
            gradients["biais3"] = np.sum(dz3, axis=0, keepdims=True)

            # ── Couche cachée 2 ──
            da2 = np.dot(dz3, self.poids3.T)
            if self.masque_dropout2 is not None:
                da2 = da2 * self.masque_dropout2 / (1.0 - self.taux_dropout)
            dz2 = da2 * self.relu_derivee(self.z2)
            gradients["poids2"] = np.dot(self.a1.T, dz2)
            gradients["biais2"] = np.sum(dz2, axis=0, keepdims=True)

            # ── Couche cachée 1 ──
            da1 = np.dot(dz2, self.poids2.T)
            if self.masque_dropout1 is not None:
                da1 = da1 * self.masque_dropout1 / (1.0 - self.taux_dropout)
            dz1 = da1 * self.relu_derivee(self.z1)
            gradients["poids1"] = np.dot(X.T, dz1)
            gradients["biais1"] = np.sum(dz1, axis=0, keepdims=True)

        # ── Mise à jour Adam ──
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr_t = self.taux_apprentissage * np.sqrt(1 - beta2 ** self._t) / (1 - beta1 ** self._t)

        params_dict = dict(self._params())
        for nom, grad in gradients.items():
            self._m[nom] = beta1 * self._m[nom] + (1 - beta1) * grad
            self._v[nom] = beta2 * self._v[nom] + (1 - beta2) * grad ** 2
            params_dict[nom] -= lr_t * self._m[nom] / (np.sqrt(self._v[nom]) + eps)

        # Ré-assigner (numpy arrays sont mutés in-place via -= mais on s'assure de la cohérence)
        self.poids1 = params_dict["poids1"]
        self.biais1 = params_dict["biais1"]
        self.poids2 = params_dict["poids2"]
        self.biais2 = params_dict["biais2"]
        self.poids3 = params_dict["poids3"]
        self.biais3 = params_dict["biais3"]
        if self.trois_couches:
            self.poids4 = params_dict["poids4"]
            self.biais4 = params_dict["biais4"]

    # ─────────────────────────────────────────────────
    # Entraînement
    # ─────────────────────────────────────────────────

    def entrainer(self, X, y, epochs=1000, taille_batch=32, afficher_progression=True):
        """
        Entraîne le réseau sur les données avec mini-batch Adam.

        :param X:                    Données d'entrée (n_echantillons, taille_entree)
        :param y:                    Labels one-hot (n_echantillons, n_classes)
        :param epochs:               Nombre d'epochs d'entraînement
        :param taille_batch:         Taille des mini-batchs (0 = batch complet)
        :param afficher_progression: Affiche la perte/précision toutes les 100 epochs
        """
        n = X.shape[0]
        batch = taille_batch if taille_batch > 0 else n

        for epoch in range(epochs):
            # Mélange des données à chaque epoch
            indices = np.random.permutation(n)
            X_shuffle = X[indices]
            y_shuffle = y[indices]

            # Mini-batchs
            for debut in range(0, n, batch):
                fin = min(debut + batch, n)
                X_batch = X_shuffle[debut:fin]
                y_batch = y_shuffle[debut:fin]
                self.forward(X_batch, entrainement=True)
                self.backward(X_batch, y_batch)

            # Affichage de la progression (évaluation sur tout le set, sans dropout)
            if afficher_progression and (epoch + 1) % 100 == 0:
                y_pred = self.forward(X, entrainement=False)
                perte = self.calculer_perte(y_pred, y)
                predictions = np.argmax(y_pred, axis=1)
                labels_reels = np.argmax(y, axis=1)
                precision = np.mean(predictions == labels_reels) * 100
                print(f"Epoch {epoch + 1:5d}/{epochs} — Perte: {perte:.6f} — Précision: {precision:.1f}%")

    # ─────────────────────────────────────────────────
    # Prédiction
    # ─────────────────────────────────────────────────

    def predire(self, X):
        """
        Prédit la classe pour un vecteur d'entrée X (sans dropout).

        :param X: Vecteur d'entrée (1, taille_entree)
        :return:  (indice_classe, confiance)
        """
        probabilites = self.forward(X, entrainement=False)
        indice = int(np.argmax(probabilites))
        confiance = float(probabilites[0][indice])
        return indice, confiance

    # ─────────────────────────────────────────────────
    # Sauvegarde / Chargement des poids
    # ─────────────────────────────────────────────────

    def sauvegarder(self, chemin="brain/model.json"):
        """Sauvegarde tous les poids et biais dans un fichier JSON."""
        modele = {
            "version": self.VERSION,
            "architecture": {
                "taille_entree": self.taille_entree,
                "taille_cachee1": self.taille_cachee1,
                "taille_cachee2": self.taille_cachee2,
                "taille_cachee3": self.taille_cachee3,
                "taille_sortie": self.taille_sortie,
                "taux_apprentissage": self.taux_apprentissage,
                "taux_dropout": self.taux_dropout,
            },
            "poids": {
                "poids1": self.poids1.tolist(),
                "biais1": self.biais1.tolist(),
                "poids2": self.poids2.tolist(),
                "biais2": self.biais2.tolist(),
                "poids3": self.poids3.tolist(),
                "biais3": self.biais3.tolist(),
            },
        }
        if self.trois_couches:
            modele["poids"]["poids4"] = self.poids4.tolist()
            modele["poids"]["biais4"] = self.biais4.tolist()

        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(modele, f, ensure_ascii=False, indent=2)
        print(f"✅ Modèle v{self.VERSION} sauvegardé dans '{chemin}'")

    def charger(self, chemin="brain/model.json"):
        """Charge les poids depuis un fichier JSON."""
        with open(chemin, "r", encoding="utf-8") as f:
            modele = json.load(f)

        arch = modele["architecture"]
        self.taille_entree = arch["taille_entree"]
        self.taille_cachee1 = arch["taille_cachee1"]
        self.taille_cachee2 = arch["taille_cachee2"]
        self.taille_cachee3 = arch.get("taille_cachee3", None)
        self.taille_sortie = arch["taille_sortie"]
        self.taux_apprentissage = arch["taux_apprentissage"]
        self.taux_dropout = arch.get("taux_dropout", 0.0)
        self.trois_couches = self.taille_cachee3 is not None and self.taille_cachee3 > 0

        poids = modele["poids"]
        self.poids1 = np.array(poids["poids1"])
        self.biais1 = np.array(poids["biais1"])
        self.poids2 = np.array(poids["poids2"])
        self.biais2 = np.array(poids["biais2"])
        self.poids3 = np.array(poids["poids3"])
        self.biais3 = np.array(poids["biais3"])
        if self.trois_couches and "poids4" in poids:
            self.poids4 = np.array(poids["poids4"])
            self.biais4 = np.array(poids["biais4"])
        else:
            self.poids4 = None
            self.biais4 = None

        self._init_adam()
        self._t = 0

        version = modele.get("version", "1.0")
        print(f"✅ Modèle v{version} chargé depuis '{chemin}'")
