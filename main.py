"""
Point d'entrée principal d'Atlas.
Lance : python main.py

1. Vérifie si le modèle est entraîné
2. Si non, lance l'entraînement automatiquement
3. Lance le serveur Flask
"""

import os


CHEMIN_MODELE = os.path.join("brain", "model.json")


def main():
    print("==============================================")
    print("      🤖 Atlas — Démarrage")
    print("==============================================")

    # Vérifier si le modèle entraîné existe
    if not os.path.isfile(CHEMIN_MODELE):
        print("⚠️  Modèle non trouvé. Lancement de l'entraînement...")
        print()

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

        print()
        print("✅ Entraînement terminé !")
    else:
        print(f"✅ Modèle trouvé : {CHEMIN_MODELE}")

    print()
    print("🚀 Lancement du serveur Flask...")
    print("==============================================")

    # Lancer le serveur Flask
    from server.app import app

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Atlas démarre sur le port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
