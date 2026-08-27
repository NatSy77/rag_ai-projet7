import os
from pathlib import Path

import faiss
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral


# Chemins des fichiers générés lors des étapes précédentes.
# Fichier contenant les chunks et les métadonnées
# associées à chaque événement.
DATA_PATH = Path("data/event_chunks.csv")

# Index FAISS construit à partir des embeddings
# des différents chunks.
FAISS_INDEX_PATH = Path("data/chunks.index")

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


def search_events(
    query,
    top_k=TOP_K,
    df=None,
    index=None,
    client=None,
):
    """
    Recherche les événements les plus proches sémantiquement
    d'une requête utilisateur.

    Les dépendances peuvent être fournies directement pour
    faciliter les tests unitaires. Lors d'une utilisation
    normale, elles sont chargées automatiquement.

    Parameters
    ----------
    query : str
        Question ou besoin exprimé par l'utilisateur.

    top_k : int
        Nombre d'événements différents à retourner.

    df : pandas.DataFrame, optional
        Métadonnées des chunks. Si None, le fichier local
        event_chunks.csv est chargé.

    index : faiss.Index, optional
        Index FAISS. Si None, l'index local chunks.index
        est chargé.

    client : Mistral, optional
        Client Mistral. Si None, il est initialisé depuis .env.

    Returns
    -------
    pandas.DataFrame
        Événements les plus proches avec leur score
        de similarité.
    """

    # Chargement automatique des métadonnées en utilisation réelle.
    if df is None:
        df = pd.read_csv(DATA_PATH)

    # Chargement automatique de l'index FAISS en utilisation réelle.
    if index is None:
        index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

    # Vérification de cohérence entre les chunks
    # et les vecteurs présents dans FAISS.
    if len(df) != index.ntotal:
        raise ValueError(
            "Le nombre de chunks ne correspond pas "
            "au nombre de vecteurs présents dans l'index FAISS."
        )

    # Initialisation du client Mistral uniquement si nécessaire.
    if client is None:
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

    # On récupère davantage de chunks que le nombre final
    # d'événements souhaités, car plusieurs chunks peuvent
    # appartenir au même événement.
    search_k = min(top_k * 5, index.ntotal)

    scores, indices = index.search(
        query_vector,
        search_k,
    )

    # Construction des résultats à partir des chunks retrouvés.
    results = df.iloc[indices[0]].copy()

    # Ajout du score de similarité FAISS.
    results["similarity_score"] = scores[0]

    # Plusieurs chunks peuvent appartenir au même événement.
    # On conserve uniquement le chunk ayant obtenu le meilleur
    # score pour chaque UID.
    results = (
        results
        .sort_values(
            by="similarity_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "title_fr",
                "daterange_fr",
                "location_name",
            ],
            keep="first",
            )
        .head(top_k)
        .reset_index(drop=True)
        )

    return results


if __name__ == "__main__":
    # Requêtes utilisées pour évaluer qualitativement la pertinence de la recherche sémantique.
    queries = [
        "Je cherche un événement autour du vin à Paris",
        "Je cherche une activité pour des enfants à Paris",
        "Je voudrais visiter un musée ou voir une exposition à Paris",
        "Je cherche un concert ou un événement musical à Paris",
    ]

    # Test de chaque scénario de recherche.
    for query in queries:
        print("\n" + "=" * 80)
        print("Requête :", query)
        print("=" * 80)

        results = search_events(
            query,
            top_k=5,
        )

        print("\nÉvénements recommandés :\n")

        # Affichage des événements uniques retournés par FAISS.
        for _, event in results.iterrows():
            print(
                f"- {event['title_fr']}\n"
                f"  Date : {event['daterange_fr']}\n"
                f"  Lieu : {event['location_name']}\n"
                f"  Score : {event['similarity_score']:.4f}\n"
                f"  Chunk : {event['chunk_text'][:200]}...\n"
            )