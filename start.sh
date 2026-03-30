#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Script de démarrage d'Atlas (délègue à main.py)
# Les erreurs sont gérées par main.py (logging dans logs/atlas.log).
# ─────────────────────────────────────────────────────────────────────────────

exec python main.py
