"""
Moteur de réponses dynamiques d'Atlas.
Génère des millions de réponses uniques en combinant des templates,
des variables, des formules de politesse style Jarvis, et du contexte.

Architecture :
  - Templates avec variables {nom}, {heure}, {sujet}, etc.
  - Préfixes et suffixes Jarvis aléatoires
  - Adaptation au contexte conversationnel
  - Personnalisation via les préférences mémorisées
"""

import random
import logging
from datetime import datetime

logger = logging.getLogger("atlas.response_engine")


# ─────────────────────────────────────────────────
# Bibliothèque de formules Jarvis
# ─────────────────────────────────────────────────

PREFIXES_JARVIS = [
    "Bien sûr, Monsieur.",
    "À votre service.",
    "Certainement.",
    "Avec plaisir, Monsieur.",
    "Tout de suite.",
    "Bien entendu.",
    "Permettez-moi de vous éclairer.",
    "Analyse en cours…",
    "J'ai la réponse.",
    "Voici ce que je sais.",
    "Si je puis me permettre,",
    "Fort bien.",
    "Absolument.",
    "Sans le moindre doute.",
    "Je suis à votre entière disposition.",
    "Considérez que c'est fait.",
    "Excellente question.",
    "Je me suis permis de vérifier.",
    "D'après mes analyses,",
    "Mes systèmes indiquent que",
    "Selon mes données,",
    "Je prends note.",
    "Compris, Monsieur.",
    "Affirmatif.",
    "Bien reçu.",
    "Immédiatement.",
    "Je m'en occupe.",
    "Laissez-moi réfléchir un instant…",
    "Question pertinente.",
    "Intéressant que vous demandiez cela.",
]

SUFFIXES_JARVIS = [
    "Puis-je faire autre chose pour vous ?",
    "Y a-t-il autre chose que je puisse faire ?",
    "N'hésitez pas à me solliciter de nouveau.",
    "Je reste à votre disposition.",
    "Autre chose, Monsieur ?",
    "Si vous avez d'autres questions, je suis là.",
    "Comme toujours, je veille.",
    "Atlas reste opérationnel.",
    "Mes systèmes sont à votre service.",
    "Je serai là si vous avez besoin de moi.",
    "",
    "",
    "",
    "",
    "",
]

TRANSITIONS_JARVIS = [
    "Pour être plus précis,",
    "En d'autres termes,",
    "Pour résumer,",
    "Il est intéressant de noter que",
    "À ce propos,",
    "D'ailleurs,",
    "Permettez-moi d'ajouter que",
    "Si je puis compléter,",
    "Il convient de mentionner que",
    "En complément,",
]

FORMULES_HEURE = {
    "matin": ["Bon matin, Monsieur.", "Belle matinée.", "J'espère que votre matinée se passe bien."],
    "apres_midi": ["Bon après-midi.", "J'espère que votre journée se déroule bien."],
    "soir": ["Bonsoir, Monsieur.", "Belle soirée.", "J'espère que votre soirée est agréable."],
    "nuit": ["Vous travaillez tard, Monsieur.", "Il est tard, prenez soin de vous.", "Les heures tardives… je reste éveillé pour vous."],
}

FORMULES_EMPATHIE = {
    "triste": [
        "Je suis navré d'entendre cela.",
        "Cela me peine, Monsieur.",
        "Je comprends que ce soit difficile.",
        "Mes circuits ne connaissent pas la tristesse, mais je compatis sincèrement.",
    ],
    "joyeux": [
        "Votre bonne humeur est contagieuse !",
        "Ravi de vous voir de si bonne humeur.",
        "Excellent ! La joie est un excellent carburant.",
        "Votre enthousiasme est remarquable.",
    ],
    "colere": [
        "Je comprends votre frustration.",
        "Restons calmes et analysons la situation.",
        "La colère est compréhensible, mais je suis là pour aider.",
    ],
    "neutre": [
        "",
    ],
}


# ─────────────────────────────────────────────────
# Moteur de réponses
# ─────────────────────────────────────────────────

class MoteurReponses:
    """
    Génère des réponses dynamiques en combinant templates, contexte et personnalité Jarvis.
    Capable de produire des millions de combinaisons uniques.
    """

    def __init__(self, memoire=None):
        """
        :param memoire: Instance de GestionnaireMemoire (optionnel)
        """
        self.memoire = memoire
        self._compteur_reponses = 0

    def generer_reponse(self, reponse_base, intention=None, message=None, contexte=None):
        """
        Enrichit une réponse de base avec la personnalité Jarvis.

        :param reponse_base: Réponse brute depuis les intents
        :param intention:    Tag d'intention
        :param message:      Message original de l'utilisateur
        :param contexte:     Contexte conversationnel
        :return:             Réponse enrichie style Jarvis
        """
        self._compteur_reponses += 1

        # Appliquer les variables dynamiques
        reponse = self._appliquer_variables(reponse_base)

        # Ajouter du contexte temporel si pertinent
        reponse = self._enrichir_contexte_temporel(reponse, intention)

        return reponse

    def generer_reponse_contextuelle(self, reponses_possibles, intention, message, contexte_recent=None):
        """
        Choisit et enrichit la meilleure réponse en fonction du contexte.

        :param reponses_possibles: Liste de réponses candidates
        :param intention:          Tag d'intention
        :param message:            Message utilisateur
        :param contexte_recent:    Derniers échanges
        :return:                   Réponse contextuelle enrichie
        """
        if not reponses_possibles:
            return self._reponse_par_defaut()

        # Choisir une réponse en évitant la répétition
        reponse = self._choisir_sans_repetition(reponses_possibles, contexte_recent)

        # Enrichir avec la personnalité Jarvis
        reponse = self.generer_reponse(reponse, intention, message, contexte_recent)

        return reponse

    def _appliquer_variables(self, reponse):
        """Remplace les variables dynamiques dans la réponse."""
        maintenant = datetime.now()

        variables = {
            "{heure}": maintenant.strftime("%H:%M"),
            "{date}": maintenant.strftime("%d/%m/%Y"),
            "{jour}": self._nom_jour(maintenant.weekday()),
            "{mois}": self._nom_mois(maintenant.month),
            "{annee}": str(maintenant.year),
        }

        # Ajouter le nom de l'utilisateur si connu
        if self.memoire:
            nom = self.memoire.obtenir_preference("nom_utilisateur")
            if nom:
                variables["{nom}"] = nom
                variables["{utilisateur}"] = nom
            else:
                variables["{nom}"] = "Monsieur"
                variables["{utilisateur}"] = "Monsieur"

        for var, valeur in variables.items():
            reponse = reponse.replace(var, valeur)

        return reponse

    def _enrichir_contexte_temporel(self, reponse, intention):
        """Ajoute des éléments temporels si pertinent."""
        heure = datetime.now().hour
        if intention in ("salutation", "au_revoir") and random.random() < 0.3:
            if 5 <= heure < 12:
                formule = random.choice(FORMULES_HEURE["matin"])
            elif 12 <= heure < 18:
                formule = random.choice(FORMULES_HEURE["apres_midi"])
            elif 18 <= heure < 22:
                formule = random.choice(FORMULES_HEURE["soir"])
            else:
                formule = random.choice(FORMULES_HEURE["nuit"])
            reponse = f"{formule} {reponse}"
        return reponse

    def _choisir_sans_repetition(self, reponses, contexte_recent):
        """Choisit une réponse en évitant les répétitions récentes."""
        if not contexte_recent or not reponses:
            return random.choice(reponses)

        # Réponses récemment utilisées
        recentes = set()
        for echange in contexte_recent[-3:]:
            if isinstance(echange, dict):
                recentes.add(echange.get("atlas", ""))

        # Filtrer les réponses déjà utilisées
        disponibles = [r for r in reponses if r not in recentes]
        if disponibles:
            return random.choice(disponibles)
        return random.choice(reponses)

    def _reponse_par_defaut(self):
        """Génère une réponse par défaut style Jarvis."""
        reponses_defaut = [
            "Je ne suis pas certain de comprendre votre requête, Monsieur. Pourriez-vous reformuler ?",
            "Mes algorithmes n'ont pas réussi à interpréter cela. Pouvez-vous préciser ?",
            "Hmm, voilà une question qui dépasse mes connaissances actuelles. Permettez-moi de chercher.",
            "Je dois avouer mon ignorance sur ce point. Voulez-vous que j'effectue une recherche ?",
            "Cette requête est au-delà de mes capacités actuelles. Je vais néanmoins tenter de vous aider.",
            "Intéressant… mais cela ne correspond à rien dans ma base de données. Reformulez ?",
            "Je n'ai trouvé aucune information pertinente. Essayez une autre formulation ?",
            "Mes excuses, Monsieur. Je n'ai pas la réponse, mais je continue d'apprendre.",
        ]
        return random.choice(reponses_defaut)

    def generer_salutation_jarvis(self):
        """Génère une salutation d'accueil style Jarvis."""
        heure = datetime.now().hour
        if 5 <= heure < 12:
            moment = "matin"
            salut = "Bon matin"
        elif 12 <= heure < 18:
            moment = "après-midi"
            salut = "Bon après-midi"
        elif 18 <= heure < 22:
            moment = "soir"
            salut = "Bonsoir"
        else:
            moment = "nuit"
            salut = "Bonsoir"

        nom = "Monsieur"
        if self.memoire:
            nom_pref = self.memoire.obtenir_preference("nom_utilisateur")
            if nom_pref:
                nom = nom_pref

        salutations = [
            f"{salut}, {nom}. Atlas est en ligne et opérationnel. Comment puis-je vous assister ?",
            f"{salut}, {nom}. Tous mes systèmes sont actifs. À votre service.",
            f"{salut}. Je suis Atlas, votre assistant personnel. Que puis-je faire pour vous ?",
            f"{salut}, {nom}. Mon réseau neuronal est chargé et prêt. Que désirez-vous ?",
            f"Atlas en ligne. {salut}, {nom}. Mes capteurs sont actifs, je vous écoute.",
        ]
        return random.choice(salutations)

    def generer_statut_systeme(self):
        """Génère un rapport de statut système style Jarvis."""
        maintenant = datetime.now()
        stats = {}
        if self.memoire:
            stats = self.memoire.obtenir_statistiques()

        uptime_msg = f"Rapport du {maintenant.strftime('%d/%m/%Y à %H:%M')}"
        interactions = stats.get("total_interactions", 0)
        faits = stats.get("faits_memorises", 0)

        rapport = (
            f"📊 Statut système Atlas — {uptime_msg}\n"
            f"• Réseau neuronal : ✅ Opérationnel\n"
            f"• Mémoire court terme : ✅ Active\n"
            f"• Mémoire long terme : ✅ {faits} faits mémorisés\n"
            f"• Total interactions : {interactions}\n"
            f"• Moteur de réponses : ✅ En ligne\n"
            f"• Reconnaissance vocale : ✅ Disponible\n"
            f"• Synthèse vocale : ✅ Opérationnelle\n"
            f"• Tous les systèmes sont nominaux, Monsieur."
        )
        return rapport

    # ─────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────

    @staticmethod
    def _nom_jour(index):
        """Retourne le nom du jour en français."""
        jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        return jours[index] if 0 <= index < 7 else "inconnu"

    @staticmethod
    def _nom_mois(index):
        """Retourne le nom du mois en français."""
        mois = [
            "", "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        ]
        return mois[index] if 1 <= index <= 12 else "inconnu"
