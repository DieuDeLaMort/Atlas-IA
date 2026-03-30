"""
Tokenizer maison pour Atlas.
Implémente :
  - Nettoyage et tokenisation du texte
  - Stemming basique (suffixes français et anglais)
  - Vectorisation Bag of Words (BoW)
  - Sauvegarde / chargement du vocabulaire
"""

import json
import re


class Tokenizer:
    """
    Tokenizer entièrement codé from scratch — pas de NLTK, pas de spaCy.
    Supporte le français et l'anglais.
    """

    # Suffixes à supprimer pour le stemming français
    SUFFIXES_FR = [
        "aient", "assent", "issent", "ussent",
        "eras", "erai", "erez", "erons", "eront",
        "ais", "ait", "ions", "iez", "aient",
        "tion", "sion", "ment", "ments",
        "eur", "eurs", "euse", "euses",
        "ique", "iques", "iste", "istes",
        "ant", "ants", "ante", "antes",
        "er", "ir", "re", "ez", "es", "ent",
        "ons", "ais", "ait",
    ]

    # Suffixes à supprimer pour le stemming anglais
    SUFFIXES_EN = [
        "ational", "tional", "enci", "anci", "izer",
        "ising", "izing", "ation", "ations", "ator",
        "alism", "ness", "ment", "ments",
        "ing", "ings", "tion", "tions",
        "ies", "ness", "ful", "less",
        "ly", "ed", "er", "es", "s",
    ]

    def __init__(self):
        self.vocabulaire = []      # Liste des tokens connus
        self.vocab_index = {}      # {token: indice}

    # ─────────────────────────────────────────────────
    # Nettoyage du texte
    # ─────────────────────────────────────────────────

    def nettoyer(self, texte):
        """
        Met en minuscules et supprime la ponctuation et les caractères spéciaux.

        :param texte: Chaîne d'entrée
        :return:      Chaîne nettoyée
        """
        texte = texte.lower()
        # Remplacer les caractères accentués courants pour faciliter le matching
        remplacements = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'î': 'i', 'ï': 'i',
            'ô': 'o', 'ö': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n',
        }
        for accent, sans_accent in remplacements.items():
            texte = texte.replace(accent, sans_accent)
        # Supprimer tout ce qui n'est pas une lettre ou un espace
        texte = re.sub(r"[^a-z\s]", " ", texte)
        # Normaliser les espaces multiples
        texte = re.sub(r"\s+", " ", texte).strip()
        return texte

    # ─────────────────────────────────────────────────
    # Tokenisation
    # ─────────────────────────────────────────────────

    def tokeniser(self, texte):
        """
        Découpe le texte en une liste de tokens (mots).

        :param texte: Chaîne d'entrée
        :return:      Liste de tokens
        """
        texte_propre = self.nettoyer(texte)
        tokens = texte_propre.split()
        return tokens

    # ─────────────────────────────────────────────────
    # Stemming basique
    # ─────────────────────────────────────────────────

    def stemmer(self, mot):
        """
        Applique un stemming basique en supprimant les suffixes connus.
        Essaie d'abord les suffixes français, puis anglais.

        :param mot: Mot à « stemmer »
        :return:    Racine du mot
        """
        # On ne stemme pas les mots trop courts
        if len(mot) <= 3:
            return mot

        # Essai suffixes français (du plus long au plus court)
        for suffixe in sorted(self.SUFFIXES_FR, key=len, reverse=True):
            if mot.endswith(suffixe) and len(mot) - len(suffixe) >= 3:
                return mot[: -len(suffixe)]

        # Essai suffixes anglais (du plus long au plus court)
        for suffixe in sorted(self.SUFFIXES_EN, key=len, reverse=True):
            if mot.endswith(suffixe) and len(mot) - len(suffixe) >= 3:
                return mot[: -len(suffixe)]

        return mot

    def tokeniser_et_stemmer(self, texte):
        """
        Tokenise le texte puis applique le stemming à chaque token.

        :param texte: Chaîne d'entrée
        :return:      Liste de racines de tokens
        """
        tokens = self.tokeniser(texte)
        return [self.stemmer(t) for t in tokens]

    # ─────────────────────────────────────────────────
    # Construction du vocabulaire
    # ─────────────────────────────────────────────────

    def construire_vocabulaire(self, phrases):
        """
        Construit le vocabulaire à partir d'une liste de phrases.

        :param phrases: Liste de chaînes de caractères
        """
        tokens_uniques = set()
        for phrase in phrases:
            for token in self.tokeniser_et_stemmer(phrase):
                tokens_uniques.add(token)

        self.vocabulaire = sorted(tokens_uniques)
        self.vocab_index = {token: i for i, token in enumerate(self.vocabulaire)}

    # ─────────────────────────────────────────────────
    # Vectorisation Bag of Words
    # ─────────────────────────────────────────────────

    def vectoriser(self, texte):
        """
        Transforme une phrase en vecteur Bag of Words.

        :param texte: Chaîne d'entrée
        :return:      Liste de flottants de taille len(vocabulaire) (1.0 si le token est présent, 0.0 sinon)
        """
        vecteur = [0.0] * len(self.vocabulaire)
        tokens = self.tokeniser_et_stemmer(texte)
        for token in tokens:
            if token in self.vocab_index:
                vecteur[self.vocab_index[token]] = 1.0
        return vecteur

    # ─────────────────────────────────────────────────
    # Sauvegarde / Chargement
    # ─────────────────────────────────────────────────

    def sauvegarder(self, chemin="data/vocabulary.json"):
        """
        Sauvegarde le vocabulaire dans un fichier JSON.

        :param chemin: Chemin du fichier de sauvegarde
        """
        donnees = {
            "vocabulaire": self.vocabulaire,
            "vocab_index": self.vocab_index
        }
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)
        print(f"✅ Vocabulaire ({len(self.vocabulaire)} tokens) sauvegardé dans '{chemin}'")

    def charger(self, chemin="data/vocabulary.json"):
        """
        Charge le vocabulaire depuis un fichier JSON.

        :param chemin: Chemin du fichier de sauvegarde
        """
        with open(chemin, "r", encoding="utf-8") as f:
            donnees = json.load(f)

        self.vocabulaire = donnees["vocabulaire"]
        self.vocab_index = {token: int(i) for token, i in donnees["vocab_index"].items()}
        print(f"✅ Vocabulaire ({len(self.vocabulaire)} tokens) chargé depuis '{chemin}'")
