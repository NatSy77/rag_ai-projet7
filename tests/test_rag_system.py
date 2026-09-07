import pandas as pd
import pytest

from scripts.rag_system import RAGSystem


class FakeIndex:
    """
    Faux index FAISS utilisé pour tester RAGSystem
    sans charger l'index réel.
    """

    def __init__(self, ntotal):
        self.ntotal = ntotal


class FakeClient:
    """
    Faux client Mistral.
    Aucun appel API réel n'est effectué.
    """

    pass


def test_rag_system_rejects_inconsistent_index():
    """
    Vérifie qu'une erreur est levée lorsque le nombre
    de chunks ne correspond pas au nombre de vecteurs FAISS.
    """

    df = pd.DataFrame(
        [
            {"uid": "1"},
            {"uid": "2"},
        ]
    )

    index = FakeIndex(ntotal=3)

    with pytest.raises(
        ValueError,
        match="Le nombre de chunks ne correspond pas",
    ):
        RAGSystem(
            client=FakeClient(),
            chunks_df=df,
            index=index,
        )


def test_build_context_contains_event_information():
    """
    Vérifie que le contexte transmis au LLM contient
    les informations provenant des événements récupérés.
    """

    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title_fr": "Concert de jazz",
                "daterange_fr": "10 septembre 2026",
                "location_name": "Salle de concert",
                "chunk_text": "Concert de jazz à Paris.",
                "canonicalurl": "https://example.com/concert",
                "similarity_score": 0.92,
            }
        ]
    )

    rag = RAGSystem(
        client=FakeClient(),
        chunks_df=df,
        index=FakeIndex(ntotal=1),
        generation_chain=FakeGenerationChain(),
    )

    context = rag.build_context(df)

    assert "Concert de jazz" in context
    assert "10 septembre 2026" in context
    assert "Salle de concert" in context
    assert "Concert de jazz à Paris." in context
    assert "https://example.com/concert" in context
    
import numpy as np


class FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingsResponse:
    def __init__(self, embedding):
        self.data = [
            FakeEmbeddingItem(embedding)
        ]


class FakeEmbeddingsAPI:
    def create(self, model, inputs):
        return FakeEmbeddingsResponse(
            [1.0, 0.0]
        )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [
            FakeChoice(content)
        ]


class FakeChatAPI:
    def complete(
        self,
        model,
        messages,
        temperature,
        max_tokens,
    ):
        return FakeChatResponse(
            "Voici une recommandation de test."
        )


class FullFakeClient:
    """
    Faux client Mistral utilisé pour les tests du retrieval.
    """

    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()


class SearchFakeIndex:
    """
    Faux index FAISS renvoyant des scores
    et indices prédéfinis.
    """

    def __init__(self):
        self.ntotal = 3

    def search(self, query_vector, k):
        scores = np.array(
            [[0.95, 0.90, 0.85]],
            dtype=np.float32,
        )

        indices = np.array(
            [[0, 1, 2]],
            dtype=np.int64,
        )

        return scores, indices
    
class FakeGenerationChain:
    """
    Fausse chaîne LangChain utilisée pendant les tests.
    Aucun appel réel au LLM n'est effectué.
    """

    def invoke(self, inputs):
        return "Voici une recommandation de test."


def create_fake_chunks():
    """
    Crée un petit jeu de chunks utilisé
    pour les tests du système RAG.
    """

    return pd.DataFrame(
        [
            {
                "uid": "1",
                "title_fr": "Concert de jazz",
                "daterange_fr": "10 septembre 2026",
                "lastdate_end": "2026-09-10T22:00:00+00:00",
                "location_name": "Salle A",
                "chunk_text": "Concert de jazz à Paris.",
                "canonicalurl": "https://example.com/1",
            },
            {
                "uid": "2",
                "title_fr": "Exposition",
                "daterange_fr": "11 septembre 2026",
                "lastdate_end": "2026-09-10T22:00:00+00:00",
                "location_name": "Musée B",
                "chunk_text": "Exposition culturelle à Paris.",
                "canonicalurl": "https://example.com/2",
            },
            {
                "uid": "3",
                "title_fr": "Atelier enfants",
                "daterange_fr": "12 septembre 2026",
                "lastdate_end": "2026-09-10T22:00:00+00:00",
                "location_name": "Lieu C",
                "chunk_text": "Atelier culturel pour enfants.",
                "canonicalurl": "https://example.com/3",
            },
        ]
    )


def test_retrieve_returns_ranked_results():
    """
    Vérifie que retrieve() retourne les résultats
    accompagnés de leurs scores de similarité.
    """

    df = create_fake_chunks()

    rag = RAGSystem(
        client=FullFakeClient(),
        chunks_df=df,
        index=SearchFakeIndex(),
        generation_chain=FakeGenerationChain(),
    )

    results = rag.retrieve(
        "Je cherche un concert",
        top_k=2,
    )

    assert len(results) == 2
    assert results.iloc[0]["title_fr"] == "Concert de jazz"
    assert results.iloc[0]["similarity_score"] > 0.94
    assert (
        results["similarity_score"]
        .is_monotonic_decreasing
    )


def test_ask_returns_answer_context_and_sources():
    """
    Vérifie que ask() retourne une structure directement
    réutilisable par la future API REST.
    """

    df = create_fake_chunks()

    rag = RAGSystem(
        client=FullFakeClient(),
        chunks_df=df,
        index=SearchFakeIndex(),
        generation_chain=FakeGenerationChain(),
    )

    result = rag.ask(
        "Je cherche un concert",
        top_k=2,
    )

    assert "answer" in result
    assert "context" in result
    assert "sources" in result

    assert (
        result["answer"]
        == "Voici une recommandation de test."
    )

    assert "Concert de jazz" in result["context"]

    assert len(result["sources"]) == 2

    assert (
        result["sources"][0]["title"]
        == "Concert de jazz"
    )

    assert (
        result["sources"][0]["url"]
        == "https://example.com/1"
    )
    
def test_retrieve_rejects_empty_question():
    """
    Vérifie qu'une question vide est rejetée
    avant toute recherche ou appel à l'API.
    """

    df = create_fake_chunks()

    rag = RAGSystem(
        client=FullFakeClient(),
        chunks_df=df,
        index=SearchFakeIndex(),
        generation_chain=FakeGenerationChain(),
    )

    with pytest.raises(
        ValueError,
        match="La question ne peut pas être vide",
    ):
        rag.retrieve("   ")