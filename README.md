# Atlas 🤖

**Atlas** est un chatbot IA entièrement codé *from scratch* — sans TensorFlow, sans PyTorch, sans scikit-learn. Uniquement **Python + NumPy + Flask**.

Il dispose d'une interface web moderne (thème dark) et peut être hébergé sur **Pterodactyl** via Docker.

---

## ✨ Fonctionnalités

- 🧠 Réseau de neurones multi-couches (backpropagation complète)
- 📝 Tokenizer maison avec stemming français/anglais
- 💬 Conversations en français sur de nombreux sujets :
  - Assistant général (salutations, présentation, aide...)
  - Support technique (PC lent, connexion, mots de passe...)
  - Gaming (Minecraft, lag, recommandations...)
  - Fun (blagues, fun facts, citations...)
- 🌐 Interface web dark & responsive
- 🐳 Prêt pour Pterodactyl (Dockerfile + start.sh)

---

## 🗂️ Architecture

```
Atlas-IA/
├── brain/
│   ├── __init__.py
│   ├── neural_network.py    # Réseau de neurones from scratch (NumPy)
│   ├── tokenizer.py         # Tokenizer maison (bag of words + stemming)
│   ├── trainer.py           # Entraînement du modèle
│   ├── model.json           # Poids sauvegardés (généré après train.py)
│   └── classes.json         # Liste des intents (généré après train.py)
├── data/
│   ├── intents.json         # Base de connaissances
│   └── vocabulary.json      # Vocabulaire (généré après train.py)
├── server/
│   ├── __init__.py
│   ├── app.py               # Serveur Flask + API REST
│   └── templates/
│       └── index.html       # Interface web (HTML/CSS/JS intégré)
├── train.py                 # Script d'entraînement principal
├── requirements.txt
├── Dockerfile
├── start.sh
└── README.md
```

---

## 🚀 Installation & Lancement

### Prérequis

- Python 3.10+
- pip

### 1. Cloner le repo

```bash
git clone https://github.com/DieuDeLaMort/Atlas-IA.git
cd Atlas-IA
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Entraîner Atlas

```bash
python train.py
```

L'entraînement dure quelques secondes. Il crée :
- `brain/model.json` — les poids du réseau neuronal
- `brain/classes.json` — la liste des intents
- `data/vocabulary.json` — le vocabulaire appris

### 4. Lancer le serveur

```bash
python -m server.app
```

Ou via le script de démarrage :

```bash
chmod +x start.sh
./start.sh
```

### 5. Ouvrir l'interface

Ouvre [http://localhost:5000](http://localhost:5000) dans ton navigateur et parle à Atlas ! 🤖

---

## 🐳 Déploiement sur Pterodactyl

### Option 1 — Docker

```bash
docker build -t atlas-ia .
docker run -p 5000:5000 atlas-ia
```

### Option 2 — Pterodactyl

1. Crée un serveur avec l'egg **Python Generic** ou utilise l'image Docker
2. Upload les fichiers du projet
3. Configure le port `5000` (ou via la variable `PORT`)
4. Le script `start.sh` s'occupe de tout :
   - Entraînement automatique si le modèle n'existe pas
   - Lancement du serveur Flask

Variable d'environnement disponible :
- `PORT` — port d'écoute (défaut : `5000`)

---

## 🧠 Architecture technique

### Réseau de neurones (`brain/neural_network.py`)

- Architecture : `input → hidden1 (128) → hidden2 (64) → output`
- Activation : **ReLU** pour les couches cachées, **Softmax** pour la sortie
- Perte : **Cross-Entropy**
- Optimisation : **Gradient Descent** (descente de gradient classique)
- Initialisation : **He initialization** (adaptée pour ReLU)
- Sauvegarde/chargement des poids en **JSON**

### Tokenizer (`brain/tokenizer.py`)

- Nettoyage : mise en minuscules, suppression ponctuation, gestion des accents
- Stemming basique : suppression des suffixes français et anglais
- Vectorisation **Bag of Words** (vecteur binaire)
- Sauvegarde/chargement du vocabulaire

### Base de connaissances (`data/intents.json`)

Chaque intent contient :
- `tag` — identifiant unique
- `patterns` — phrases d'exemple (entraînement)
- `responses` — réponses possibles (sélection aléatoire)

---

## ➕ Ajouter de nouvelles connaissances

1. Ouvre `data/intents.json`
2. Ajoute un nouvel intent :

```json
{
  "tag": "mon_nouveau_sujet",
  "patterns": [
    "phrase exemple 1",
    "phrase exemple 2",
    "autre façon de demander"
  ],
  "responses": [
    "Réponse A",
    "Réponse B"
  ]
}
```

3. Ré-entraîne Atlas :

```bash
python train.py
```

4. Redémarre le serveur — Atlas connaît maintenant le nouveau sujet !

---

## 🛠️ API REST

| Méthode | Route    | Description |
|---------|----------|-------------|
| `GET`   | `/`      | Interface web |
| `POST`  | `/chat`  | Envoyer un message |
| `GET`   | `/health`| Healthcheck |

### Exemple `/chat`

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bonjour Atlas !"}'
```

Réponse :
```json
{
  "response": "Salut ! Comment puis-je t'aider aujourd'hui ?"
}
```

---

## 📦 Dépendances

```
numpy      # Calcul matriciel (réseau de neurones)
flask      # Serveur web
flask-cors # CORS pour les requêtes cross-origin
```

**Aucune librairie d'IA** n'est utilisée.

---

## 📄 Licence

MIT — Libre d'utilisation et de modification.
