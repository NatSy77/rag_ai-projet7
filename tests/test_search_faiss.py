import numpy as np
import pandas as pd

from scripts.search_faiss import search_events


class FakeEmbeddings:
    """
    Simule la réponse de l'API Mistral pour éviter
    tout appel réseau pendant les tests.
    """

    def create(self, model, inputs):
        class FakeEmbedding:
            embedding = [1.0, 0.0]

        class FakeResponse:
            data = [FakeEmbedding()]

        return FakeResponse()


class FakeClient:
    """
    Faux client Mistral utilisé uniquement pour les tests.
    """

    def __init__(self):
        self.embeddings = FakeEmbeddings()


class FakeIndex:
    """
    Simule un petit index FAISS.

    Les indices retournés permettent notamment de vérifier
    la déduplication des événements.
    """

    ntotal = 5

    def search(self, query_vector, k):
        scores = np.array(
            [[0.95, 0.90, 0.85, 0.80, 0.75]],
            dtype=np.float32,
        )

        indices = np.array(
            [[0, 1, 2, 3, 4]],
            dtype=np.int64,
        )

        return scores, indices


def test_search_events_deduplicates_and_respects_top_k():
    """
    Vérifie que la recherche supprime les doublons métier
    et respecte le nombre de résultats demandé.
    """

    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title_fr": "Concert de jazz",
                "daterange_fr": "10 septembre",
                "location_name": "Paris",
                "chunk_text": "Concert de jazz à Paris",
            },
            {
                # Même événement métier mais UID différent.
                "uid": "2",
                "title_fr": "Concert de jazz",
                "daterange_fr": "10 septembre",
                "location_name": "Paris",
                "chunk_text": "Jazz et musique à Paris",
            },
            {
                "uid": "3",
                "title_fr": "Exposition",
                "daterange_fr": "11 septembre",
                "location_name": "Musée",
                "chunk_text": "Exposition dans un musée",
            },
            {
                "uid": "4",
                "title_fr": "Atelier enfants",
                "daterange_fr": "12 septembre",
                "location_name": "Bibliothèque",
                "chunk_text": "Atelier pour enfants",
            },
            {
                "uid": "5",
                "title_fr": "Visite guidée",
                "daterange_fr": "13 septembre",
                "location_name": "Paris",
                "chunk_text": "Visite guidée de Paris",
            },
        ]
    )

    results = search_events(
        query="Je cherche un concert",
        top_k=3,
        df=df,
        index=FakeIndex(),
        client=FakeClient(),
    )

    # La fonction doit retourner exactement trois événements.
    assert len(results) == 3

    # Le doublon "Concert de jazz" doit avoir été supprimé.
    assert results["title_fr"].tolist().count(
        "Concert de jazz"
    ) == 1

    # Le meilleur score doit être conservé.
    assert results.iloc[0]["similarity_score"] > 0.94

    # Les résultats doivent être classés du meilleur
    # score au moins bon.
    assert results["similarity_score"].is_monotonic_decreasing