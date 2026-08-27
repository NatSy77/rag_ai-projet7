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
├── data/                 # Données et artefacts générés localement
├── scripts/              # Scripts de récupération, traitement et indexation
├── tests/                # Tests unitaires
├── env/                  # Environnement virtuel local (non versionné)
├── .env                  # Clé API locale (non versionnée)
├── .gitignore
├── requirements.txt
└── README.md
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

Un fichier `.env` est utilisé localement pour stocker notamment la clé API Mistral.

Le fichier `.env` est exclu du dépôt grâce au fichier `.gitignore` et ne doit jamais être versionné.

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

**Étape 2 — Pré-processing et préparation des données**

* [x] Récupération des événements OpenAgenda
* [x] Filtrage des événements sur Paris
* [x] Nettoyage et normalisation des données
* [x] Préparation du champ `text_for_embedding`
* [x] Génération et validation des embeddings
* [x] Tests unitaires

**Étape 3 — Base de données vectorielle FAISS**

* [x] Découpage des événements en chunks
* [x] Conservation des métadonnées
* [x] Vectorisation des chunks avec Mistral
* [x] Construction de l'index FAISS
* [x] Vérification du nombre de vecteurs indexés
* [x] Recherche par similarité sémantique
* [x] Déduplication des événements
* [x] Tests de plusieurs scénarios de recherche
* [x] Tests unitaires

## Remarque sur la compatibilité

L'environnement utilise actuellement :

```text
Python 3.11
NumPy 1.26.4
FAISS CPU 1.8.0
```

Ces versions ont été retenues afin d'assurer la compatibilité de FAISS avec l'environnement de développement utilisé pour le POC.

Le fichier `requirements.txt` permet de reproduire les dépendances nécessaires au projet.

## Indexation vectorielle avec FAISS

Les événements nettoyés sont indexés dans une base vectorielle FAISS afin de permettre une recherche rapide par similarité sémantique.

### Découpage des événements en chunks

Avant la vectorisation, le champ `text_for_embedding` de chaque événement est découpé en chunks avec `RecursiveCharacterTextSplitter`.

Les paramètres retenus sont :

- taille maximale d'un chunk : `1200` caractères ;
- chevauchement entre deux chunks : `200` caractères ;
- fusion du dernier chunk avec le précédent lorsqu'il contient moins de 200 caractères.

Cette stratégie permet de conserver suffisamment de contexte tout en évitant de représenter les événements les plus longs par un seul vecteur.

Après découpage :

- nombre d'événements : `9701` ;
- nombre de chunks : `15395` ;
- nombre moyen de chunks par événement : `1.59`.

### Métadonnées

Chaque chunk conserve les informations permettant de retrouver son événement d'origine, notamment :

- l'identifiant `uid` ;
- le numéro du chunk ;
- le titre ;
- la description ;
- les dates ;
- le nom et l'adresse du lieu ;
- le code postal et la ville ;
- l'URL de l'événement.

Les chunks et leurs métadonnées sont préparés dans `data/event_chunks.csv`.

### Vectorisation

Chaque chunk est transformé en vecteur sémantique avec le modèle d'embedding Mistral.

Résultat de la vectorisation :

```text
Nombre de chunks : 15395
Nombre de vecteurs : 15395
Dimension d'un vecteur : 1024
```

Un système de checkpoint permet de sauvegarder progressivement les embeddings et de reprendre la vectorisation en cas d'interruption ou d'erreur temporaire de l'API.

### Index FAISS

Les embeddings sont normalisés avec une normalisation L2 puis ajoutés dans un index `IndexFlatIP`.

Cette combinaison permet d'effectuer une recherche exacte basée sur la similarité cosinus entre la requête utilisateur et les chunks des événements.

L'index final contient :

```text
Dimension de l'index : 1024
Nombre de vecteurs : 15395
```

L'index est sauvegardé localement dans :

```text
data/chunks.index
```

### Recherche sémantique

Lors d'une recherche :

1. la requête utilisateur est transformée en embedding avec Mistral ;
2. le vecteur est normalisé ;
3. FAISS recherche les chunks les plus similaires ;
4. les métadonnées permettent de retrouver les événements correspondants ;
5. 5. les doublons métier sont supprimés à partir du titre, de la période et du lieu de l'événement ;
6. les événements les plus pertinents sont retournés avec leur score de similarité.

Plusieurs scénarios ont été testés, notamment :

- recherche d'événements autour du vin ;
- activités pour enfants ;
- musées et expositions ;
- concerts et événements musicaux.

Les résultats obtenus sont globalement cohérents avec les intentions exprimées dans les requêtes.

### Fichiers principaux

```text
scripts/chunk_events.py
scripts/generate_embeddings.py
scripts/build_faiss_index.py
scripts/search_faiss.py
```

Les fichiers volumineux générés localement (`event_chunks.csv`, embeddings et index FAISS) ne sont pas versionnés dans Git.

### Test de reproductibilité

L'installation des dépendances a été testée dans un environnement virtuel vierge avec Python 3.11.

La commande :

```bash
pip install -r requirements.txt
```

## Pré-processing des données OpenAgenda

Les données utilisées par le POC proviennent du jeu de données public OpenAgenda accessible via l'API OpenDataSoft.

### Périmètre des données

Pour ce POC, le périmètre retenu est :

- **Localisation :** Paris
- **Historique :** événements dont la dernière occurrence date de moins d'un an
- **Événements futurs :** inclus
- **Source :** OpenAgenda

Le filtre appliqué lors de la récupération est :

```text
location_city = Paris
lastdate_end >= 2025-08-23
```

### Tests unitaires

Des tests unitaires sont mis en place avec `pytest` afin de vérifier le bon fonctionnement de la récupération, du pré-processing des données et de la génération des embeddings.

Les tests de récupération et de génération des embeddings utilisent des réponses simulées afin de ne pas dépendre de la disponibilité des services externes ni de consommer inutilement l'API Mistral.

Les tests vérifient notamment :

- la récupération correcte des événements ;
- la gestion de la pagination de l'API ;
- la suppression des événements sans titre ni description ;
- la conservation des événements valides ;
- la suppression de la colonne `category` ;
- la normalisation du code pays ;
- la normalisation de la région ;
- la conversion des dates au format `datetime` ;
- la génération des embeddings par lots ;
- la conservation de l'ordre des embeddings entre les différents lots.

Les tests couvrent également l'indexation vectorielle :

- le découpage des événements en chunks ;
- la conservation des métadonnées associées aux chunks ;
- la gestion des textes vides ;
- la construction de l'index FAISS ;
- la présence de tous les vecteurs dans l'index ;
- la recherche par similarité ;
- la déduplication des événements dans les résultats ;
- le respect du nombre de résultats demandé (`top_k`).

État actuel de la suite de tests :

```text
17 tests réussis
```

Pour exécuter l'ensemble des tests depuis la racine du projet :

```bash
python -m pytest tests/ -v
```

## Vectorisation et indexation des événements

Les événements nettoyés sont préparés pour la recherche sémantique à l'aide d'une colonne `text_for_embedding`.

Cette représentation textuelle regroupe les informations utiles à la recommandation :

- titre ;
- description ;
- description détaillée nettoyée des balises HTML ;
- mots-clés ;
- dates ;
- lieu ;
- restrictions d'âge lorsqu'elles sont disponibles.

### Génération des embeddings

Les représentations vectorielles sont générées avec le modèle `mistral-embed`.

La génération est effectuée par lots afin de limiter la taille des requêtes envoyées à l'API.

Le script implémente également :

- une sauvegarde progressive des embeddings ;
- un mécanisme de reprise après interruption ;
- une gestion des limitations de débit de l'API ;
- une nouvelle tentative en cas d'erreur temporaire du service.

Les embeddings générés ont une dimension de 1024.

Script :

```bash
python scripts/generate_embeddings.py
