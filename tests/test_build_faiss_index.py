import numpy as np

from scripts.build_faiss_index import build_faiss_index


def test_build_faiss_index_contains_all_vectors():
    """
    Vérifie que tous les embeddings sont ajoutés
    dans l'index FAISS.
    """

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = build_faiss_index(embeddings)

    assert index.ntotal == 3
    assert index.d == 2


def test_build_faiss_index_returns_relevant_result():
    """
    Vérifie qu'une recherche retourne le vecteur
    le plus proche sémantiquement.
    """

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = build_faiss_index(embeddings)

    query = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    # Même normalisation que celle utilisée dans l'index.
    import faiss

    faiss.normalize_L2(query)

    scores, indices = index.search(
        query,
        1,
    )

    assert indices[0][0] == 0
    assert scores[0][0] > 0.99


def test_build_faiss_index_rejects_empty_embeddings():
    """
    Vérifie qu'une matrice vide provoque une erreur claire.
    """

    embeddings = np.empty(
        (0, 2),
        dtype=np.float32,
    )

    try:
        build_faiss_index(embeddings)

    except ValueError as error:
        assert "Aucun embedding" in str(error)

    else:
        raise AssertionError(
            "Une matrice vide aurait dû provoquer une ValueError."
        )