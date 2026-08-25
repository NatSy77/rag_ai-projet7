import os
from pathlib import Path

import faiss
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral


# Chemins des fichiers générés lors des étapes précédentes.
DATA_PATH = Path("data/events_clean.csv")
FAISS_INDEX_PATH = Path("data/events.index")

# Modèle utilisé pour vectoriser les événements.
# La requête doit utiliser le même modèle afin que les vecteurs
# soient comparables dans le même espace vectoriel.
EMBEDDING_MODEL = "mistral-embed"

# Nombre de résultats retournés par défaut.
TOP_K = 5


def get_mistral_client():
    """
    Initialise le client Mistral à partir de la clé API
    stockée dans le fichier .env.
    """
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable MISTRAL_API_KEY est absente. "
            "Vérifiez le fichier .env."
        )

    return Mistral(api_key=api_key)


def search_events(query, top_k=TOP_K):
    """
    Recherche les événements les plus proches sémantiquement
    d'une requête utilisateur.

    Parameters
    ----------
    query : str
        Question ou besoin exprimé par l'utilisateur.

    top_k : int
        Nombre d'événements similaires à retourner.

    Returns
    -------
    pandas.DataFrame
        Événements les plus proches avec leur score
        de similarité.
    """

    # Chargement des métadonnées des événements.
    df = pd.read_csv(DATA_PATH)

    # Chargement de l'index FAISS construit précédemment.
    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    # Vérification essentielle :
    # une ligne du DataFrame doit correspondre exactement
    # à un vecteur de l'index FAISS.
    if len(df) != index.ntotal:
        raise ValueError(
            "Le nombre d'événements ne correspond pas "
            "au nombre de vecteurs présents dans l'index FAISS."
        )

    client = get_mistral_client()

    # Transformation de la requête utilisateur en embedding
    # avec le même modèle que celui utilisé pour les événements.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[query],
    )

    query_embedding = response.data[0].embedding

    # FAISS attend un tableau float32 à deux dimensions :
    # (nombre_de_requêtes, dimension_embedding).
    query_vector = pd.array(
        query_embedding,
        dtype="float32",
    ).to_numpy().reshape(1, -1)

    # L'index contient des vecteurs normalisés.
    # La requête doit donc être normalisée de la même manière.
    faiss.normalize_L2(query_vector)

    # Recherche des top_k vecteurs les plus similaires.
    scores, indices = index.search(
        query_vector,
        top_k,
    )

    # Récupération des événements correspondants.
    results = df.iloc[indices[0]].copy()

    # Ajout du score de similarité FAISS.
    results["similarity_score"] = scores[0]

    return results


if __name__ == "__main__":
    # Première requête de validation de la recherche sémantique.
    query = "Je cherche un événement autour du vin à Paris"

    print("Requête :", query)

    results = search_events(
        query,
        top_k=5,
    )

    print("\nÉvénements recommandés :\n")

    for _, event in results.iterrows():
        print(
            f"- {event['title_fr']}\n"
            f"  Date : {event['daterange_fr']}\n"
            f"  Lieu : {event['location_name']}\n"
            f"  Score : {event['similarity_score']:.4f}\n"
        )