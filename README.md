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

**Étape 4 — Système RAG avec LangChain et Mistral**

* [x] Mise en place du retrieval depuis FAISS
* [x] Construction du contexte à partir des événements
* [x] Intégration du modèle Mistral pour la génération
* [x] Orchestration de la génération avec LangChain
* [x] Création d'une classe `RAGSystem` réutilisable
* [x] Retour de la réponse, du contexte et des sources
* [x] Gestion des questions vides
* [x] Test d'une requête hors périmètre
* [x] Tests unitaires du système RAG

**Étape 5 — API REST et évaluation du système RAG**

* [x] Création d'une API REST avec FastAPI
* [x] Création de la route `POST /ask`
* [x] Création de la route `POST /rebuild`
* [x] Création des routes `/` et `/health`
* [x] Validation des entrées avec Pydantic
* [x] Séparation entre la logique RAG et l'API
* [x] Ajout d'un filtrage des événements terminés
* [x] Création de tests unitaires pour l'API
* [x] Création d'un test fonctionnel de l'API
* [x] Création d'un jeu de 10 questions d'évaluation
* [x] Évaluation qualitative des réponses
* [x] Expérimentation d'une évaluation automatique avec Ragas


## API REST avec FastAPI

Le système RAG est exposé à travers une API REST développée avec **FastAPI**.

La logique métier reste encapsulée dans la classe `RAGSystem`, tandis que l'API est définie dans le dossier :

```text
app/
├── __init__.py
├── main.py
└── schemas.py
```

Cette séparation permet de réutiliser le système RAG indépendamment de l'interface utilisée.

### Lancement de l'API

Depuis la racine du projet :

```bash
uvicorn app.main:app --reload
```

L'API est alors accessible localement sur le port `8000`.

La documentation interactive Swagger est disponible sur :

```text
http://127.0.0.1:8000/docs
```

### Routes disponibles

#### `GET /`

Permet de vérifier que l'API est accessible.

#### `GET /health`

Retourne l'état de fonctionnement de l'API.

Exemple :

```json
{
  "status": "ok"
}
```

#### `POST /ask`

Permet d'envoyer une question au système RAG.

Exemple de requête :

```json
{
  "question": "Quels concerts de jazz sont disponibles à Paris ?"
}
```

La réponse contient :

* la réponse générée ;
* le contexte utilisé ;
* les événements sources ;
* leurs métadonnées ;
* leur score de similarité.

#### `POST /rebuild`

Permet de reconstruire la base vectorielle à partir des données OpenAgenda.

Le processus réalise successivement :

```text
Récupération OpenAgenda
        ↓
Pré-processing
        ↓
Découpage en chunks
        ↓
Génération des embeddings
        ↓
Construction de l'index FAISS
        ↓
Rechargement du système RAG
```

Cette opération peut être longue et consommer des appels à l'API Mistral. Elle est donc destinée principalement à l'administration du POC.


### Filtrage temporel des recommandations

Le retrieval a été amélioré afin d'éviter de recommander des événements déjà terminés.

Après la recherche FAISS, le système utilise `lastdate_end` pour conserver uniquement les événements encore en cours ou à venir.

Un nombre plus important de candidats est récupéré avant ce filtrage afin de disposer de suffisamment de résultats pertinents après suppression des événements passés et des doublons.


## Tests de l'API

Les routes FastAPI sont testées avec `pytest` et `TestClient`.

Les tests couvrent notamment :

* les routes `/` et `/health` ;
* une requête valide vers `/ask` ;
* les questions vides ou composées uniquement d'espaces ;
* les requêtes invalides ;
* le fonctionnement de `/rebuild` ;
* la gestion d'une erreur pendant la reconstruction.

Le test fonctionnel `api_test.py` permet également de tester l'API en fonctionnement réel.

Pour lancer la suite de tests unitaires :

```bash
python -m pytest tests -q
```

Résultat obtenu :

```text
30 passed
```

Un avertissement de dépréciation lié à `TestClient` et `httpx` reste présent mais n'empêche pas l'exécution des tests.


## Évaluation du système RAG

Un jeu d'évaluation de **10 questions représentatives** a été créé dans :

```text
evaluation_questions.csv
```

Il couvre plusieurs catégories :

* concerts ;
* expositions ;
* activités familiales ;
* musées ;
* spectacles ;
* événements gratuits ;
* recherches liées à une date ;
* requête hors périmètre.

Les réponses produites par le RAG ont été sauvegardées dans :

```text
evaluation_results.csv
```

Une évaluation humaine utilise trois niveaux :

* `correct` : réponse pertinente et fidèle aux informations récupérées ;
* `partial` : réponse globalement pertinente mais incomplète ou insuffisamment justifiée ;
* `incorrect` : réponse erronée, hors sujet ou non soutenue par les sources.

Résultats obtenus :

```text
Questions évaluées : 10
Correctes           : 9 (90 %)
Partielles          : 1 (10 %)
Incorrectes         : 0 (0 %)
```

La requête hors périmètre concernant la recommandation d'un restaurant japonais à Lyon a notamment permis de vérifier que le système refuse de générer une recommandation lorsque les informations nécessaires ne sont pas présentes dans son contexte.


### Évaluation avec Ragas

Une expérimentation avec **Ragas** a également été réalisée afin d'automatiser l'évaluation du système.

Deux métriques ont été préparées :

* `Faithfulness` : mesure la fidélité de la réponse au contexte récupéré ;
* `AnswerRelevancy` : mesure la pertinence de la réponse par rapport à la question.

L'intégration utilise les modèles Mistral à travers leur interface compatible OpenAI.

Le script correspondant est :

```text
evaluate_ragas.py
```

Lors des essais, l'appel réel aux métriques Ragas a été limité par le quota de l'API Mistral et a retourné une erreur HTTP `429 Rate limit exceeded`.

L'évaluation humaine sur les 10 questions représentatives a donc été conservée comme méthode principale pour le POC.

Cette limitation met également en évidence une dépendance du système à un service externe et pourra faire l'objet d'améliorations futures : gestion du rate limiting, cache, mécanisme de retry/backoff ou utilisation d'un modèle local.

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

## Système RAG avec Mistral

Le système de recommandation utilise une architecture **RAG (Retrieval-Augmented Generation)** combinant la recherche sémantique dans FAISS et un modèle de langage Mistral.

L'objectif est de générer des réponses naturelles à partir des événements réellement présents dans la base vectorielle, tout en limitant les hallucinations du modèle.

### Orchestration avec LangChain

LangChain orchestre la partie génération du système RAG à travers une chaîne LCEL composée de :

```text
ChatPromptTemplate
        ↓
ChatMistralAI
        ↓
StrOutputParser

### Architecture du système RAG

Le traitement d'une question suit les étapes suivantes :

```text
Question utilisateur
        ↓
Embedding de la question avec Mistral
        ↓
Recherche sémantique dans FAISS
        ↓
Sélection des chunks pertinents
        ↓
Récupération des métadonnées des événements
        ↓
Construction du contexte
        ↓
Prompt envoyé au LLM
        ↓
Mistral Small
        ↓
Réponse augmentée + sources
```

### Classe `RAGSystem`

La logique métier du RAG est encapsulée dans la classe :

```text
scripts/rag_system.py
```

Cette classe permet notamment de :

- charger les métadonnées et l'index FAISS une seule fois ;
- transformer la question utilisateur en embedding ;
- rechercher les événements les plus pertinents ;
- dédupliquer les résultats ;
- construire le contexte fourni au modèle de langage ;
- générer une réponse naturelle avec Mistral ;
- retourner les sources ayant servi à produire la réponse.

Cette séparation permet de rendre la logique RAG indépendante de la future API REST.

### Modèles Mistral

Deux usages distincts de Mistral sont utilisés :

```text
Embeddings : mistral-embed
Génération : mistral-small-2603
```

`mistral-embed` est utilisé pour représenter les chunks et les requêtes utilisateur sous forme de vecteurs.

`mistral-small-2603` est utilisé pour générer une réponse en langage naturel à partir du contexte récupéré dans FAISS.

### Retrieval

Pour chaque question :

1. un embedding de la question est généré ;
2. le vecteur est normalisé avec une normalisation L2 ;
3. FAISS recherche les chunks sémantiquement les plus proches ;
4. plusieurs chunks sont récupérés afin de permettre la déduplication ;
5. les meilleurs événements distincts sont conservés.

Par défaut, le système utilise les 5 événements les plus pertinents pour construire le contexte du LLM.

### Génération augmentée

Le contexte transmis au modèle contient les informations récupérées dans la base vectorielle, notamment :

- le titre de l'événement ;
- la date ;
- le lieu ;
- le chunk pertinent ;
- l'URL de l'événement.

Le prompt demande au modèle de répondre uniquement à partir du contexte fourni et de ne pas inventer d'événement, de date ou de lieu absent des données récupérées.

Si le contexte ne permet pas de répondre correctement à la question, le modèle doit indiquer qu'il ne dispose pas d'informations suffisantes.

### Structure de la réponse

La méthode `ask()` retourne une structure contenant :

```python
{
    "answer": "...",
    "context": "...",
    "sources": [
        {
            "title": "...",
            "date": "...",
            "location": "...",
            "url": "...",
            "similarity_score": 0.0
        }
    ]
}
```

Cette structure permet de conserver la réponse générée ainsi que le contexte et les sources utilisés.

Elle facilite également la future exposition du système via une API REST et l'évaluation automatique de la qualité du RAG.

### Validation qualitative

Le système a été testé manuellement avec plusieurs types de questions :

- recommandation de concerts à Paris ;
- activités pour enfants ;
- musées et expositions ;
- requête hors périmètre concernant un restaurant japonais à Lyon.

Les tests montrent que le système récupère des événements cohérents avec les demandes et génère des réponses basées sur les résultats FAISS.

Pour une demande hors périmètre, le système indique qu'il ne dispose pas des informations nécessaires plutôt que d'inventer une recommandation.

### Tests du système RAG

Les tests unitaires du RAG utilisent des clients et index simulés afin de ne pas dépendre de l'API Mistral ni de l'index FAISS réel pendant leur exécution.

Ils vérifient notamment :

- la cohérence entre le nombre de chunks et l'index FAISS ;
- la construction du contexte ;
- le retrieval et le classement des résultats ;
- la structure retournée par `ask()` ;
- la présence de la réponse, du contexte et des sources ;
- le rejet des questions vides.

État de la suite de tests à la fin de l'étape 4 :

```text
22 tests réussis
```

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

État de la suite de tests à la fin de l'étape 3 :

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
