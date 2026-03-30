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

        try:
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
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement : {e}")
            raise SystemExit(1)

        print()
        print("✅ Entraînement terminé !")
    else:
        print(f"✅ Modèle trouvé : {CHEMIN_MODELE}")

    print()
    print("🚀 Lancement du serveur Flask...")
    print("==============================================")

    # Lancer le serveur Flask
    from server.app import app

    # Détection automatique de l'IP et du port :
    # Pterodactyl définit SERVER_IP et SERVER_PORT automatiquement.
    # On accepte aussi HOST/PORT comme fallback.
    # Par défaut : 0.0.0.0 (toutes les interfaces) et port 7778.
    host = os.environ.get("SERVER_IP") or os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT") or os.environ.get("PORT", "7778"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Atlas démarre sur {host}:{port}...")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
