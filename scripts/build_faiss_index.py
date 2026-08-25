from pathlib import Path

import faiss
import numpy as np


# Fichier contenant les embeddings générés avec Mistral.
EMBEDDINGS_PATH = Path("data/embeddings.npy")

# Fichier dans lequel l'index FAISS sera sauvegardé.
FAISS_INDEX_PATH = Path("data/events.index")


def build_faiss_index(embeddings):
    """
    Construit un index FAISS à partir d'une matrice d'embeddings.

    La similarité cosinus est utilisée pour comparer les événements.
    Pour cela, les vecteurs sont d'abord normalisés puis stockés
    dans un index utilisant le produit scalaire.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Matrice de forme (nombre_evenements, dimension_embedding).

    Returns
    -------
    faiss.Index
        Index FAISS contenant les embeddings normalisés.
    """

    # FAISS attend des vecteurs au format float32.
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # Vérification de la structure de la matrice.
    if embeddings.ndim != 2:
        raise ValueError(
            "Les embeddings doivent être une matrice à deux dimensions."
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Aucun embedding disponible pour construire l'index."
        )

    # Copie afin de ne pas modifier la matrice originale.
    normalized_embeddings = embeddings.copy()

    # Normalisation L2 des vecteurs.
    # Après normalisation, le produit scalaire correspond
    # à la similarité cosinus.
    faiss.normalize_L2(normalized_embeddings)

    # Dimension d'un embedding Mistral.
    dimension = normalized_embeddings.shape[1]

    # IndexFlatIP effectue une recherche exacte par produit scalaire.
    # Pour notre volume (~9700 événements), cette solution simple
    # est largement suffisante pour le POC.
    index = faiss.IndexFlatIP(dimension)

    # Ajout de tous les événements dans l'index.
    index.add(normalized_embeddings)

    return index


if __name__ == "__main__":
    # Chargement des embeddings générés précédemment.
    embeddings = np.load(EMBEDDINGS_PATH)

    print("Dimensions des embeddings :", embeddings.shape)

    # Construction de l'index vectoriel.
    index = build_faiss_index(embeddings)

    print("Dimension de l'index :", index.d)
    print("Nombre de vecteurs dans l'index :", index.ntotal)

    # Sauvegarde de l'index sur disque.
    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    print(
        "\nIndex FAISS sauvegardé dans :",
        FAISS_INDEX_PATH,
    )