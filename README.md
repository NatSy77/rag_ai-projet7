# Assistant intelligent de recommandation d'événements culturels

## Présentation du projet

Ce projet est réalisé dans le cadre de la formation **AI Engineer d'OpenClassrooms**.

L'objectif est de développer un **Proof of Concept (POC)** d'un assistant intelligent permettant de répondre à des questions concernant des événements culturels.

Le système s'appuiera sur une architecture **RAG (Retrieval-Augmented Generation)** combinant :

* des données d'événements culturels issues d'**Open Agenda** ;
* **LangChain** pour orchestrer le système RAG ;
* des **embeddings** pour représenter les événements sous forme vectorielle ;
* **FAISS** pour la recherche vectorielle ;
* **Mistral** pour générer les réponses en langage naturel ;
* une **API REST** pour exposer le système.

## Structure du projet

La structure du dépôt sera complétée progressivement au cours du développement.

```text
rag_ai-projet7/
├── env/                  # Environnement virtuel local (non versionné)
├── .gitignore            # Fichiers et dossiers exclus de Git
├── requirements.txt      # Dépendances Python du projet
└── README.md             # Documentation du projet
```

> Le dossier `env/` est utilisé uniquement en local et ne doit jamais être ajouté au dépôt Git.

## Environnement de développement

Le projet est actuellement développé avec :

* Python 3.11
* NumPy 1.26.4
* FAISS CPU 1.8.0
* LangChain
* Hugging Face / Sentence Transformers
* Mistral AI

### Création de l'environnement virtuel

Depuis la racine du projet :

```bash
python3.11 -m venv env
```

Activer l'environnement sous macOS/Linux :

```bash
source env/bin/activate
```

### Installation des dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Vérification de l'installation

Les principaux composants peuvent être vérifiés avec les imports suivants :

```python
import faiss

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from mistralai.client import Mistral

print("Environnement RAG opérationnel")
```

Les imports utilisés correspondent à l'organisation actuelle des bibliothèques LangChain et Mistral.

## Variables d'environnement

Les clés API et autres informations sensibles ne doivent jamais être enregistrées directement dans le code ou versionnées sur GitHub.

Un fichier `.env` pourra être utilisé localement pour stocker notamment la future clé API Mistral.

Le fichier `.env` est exclu du dépôt grâce au fichier `.gitignore`.

## État actuel du projet

**Étape 1 — Configuration de l'environnement de développement**

* [x] Création de l'environnement virtuel
* [x] Installation de FAISS CPU
* [x] Installation de LangChain
* [x] Installation des embeddings Hugging Face
* [x] Installation du SDK Mistral
* [x] Vérification des principaux imports
* [x] Création du fichier `requirements.txt`
* [x] Création du fichier `.gitignore`
* [x] Création du README
* [x] Test de reproductibilité de l'environnement depuis `requirements.txt`

## Remarque sur la compatibilité

L'environnement utilise actuellement :

```text
Python 3.11
NumPy 1.26.4
FAISS CPU 1.8.0
```

Ces versions ont été retenues afin d'assurer la compatibilité de FAISS avec l'environnement de développement utilisé pour le POC.

Le fichier `requirements.txt` permet de reproduire les dépendances nécessaires au projet.

### Test de reproductibilité

L'installation des dépendances a été testée dans un environnement virtuel vierge avec Python 3.11.

La commande :

```bash
pip install -r requirements.txt
