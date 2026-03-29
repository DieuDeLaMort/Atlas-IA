"""
Script principal d'entraînement d'Atlas.
Lance : python train.py
"""

from brain.trainer import Trainer

if __name__ == "__main__":
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
