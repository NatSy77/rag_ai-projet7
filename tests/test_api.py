"""
Tests unitaires de l'API FastAPI.

Le système RAG réel est remplacé par un faux système
afin d'éviter les appels à Mistral et à FAISS pendant les tests.
"""

from fastapi.testclient import TestClient

from app.main import app
import app.main as main_module


class FakeRAGSystem:
    """Faux système RAG utilisé uniquement pendant les tests."""

    def ask(self, question):
        return {
            "answer": "Voici une réponse de test.",
            "context": "Contexte culturel de test.",
            "sources": [
                {
                    "title": "Concert de jazz",
                    "date": "10 septembre 2026",
                    "location": "Paris",
                    "url": "https://example.com/event",
                    "similarity_score": 0.95,
                }
            ],
        }


client = TestClient(app)


def test_root():
    """Vérifie que la racine de l'API répond correctement."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "API RAG événements culturels opérationnelle"
    }


def test_health():
    """Vérifie le endpoint de contrôle de santé."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_rag_response(monkeypatch):
    """Vérifie qu'une question valide retourne une réponse RAG."""

    monkeypatch.setattr(
        main_module,
        "rag_system",
        FakeRAGSystem(),
    )

    response = client.post(
        "/ask",
        json={
            "question": "Quels concerts de jazz sont disponibles à Paris ?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Voici une réponse de test."
    assert data["context"] == "Contexte culturel de test."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Concert de jazz"
    assert data["sources"][0]["similarity_score"] == 0.95
    
def test_ask_rejects_empty_question():
    """Vérifie qu'une question vide est rejetée par Pydantic."""

    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_rejects_missing_question():
    """Vérifie qu'une requête sans question est rejetée."""

    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_rejects_whitespace_question(monkeypatch):
    """
    Vérifie qu'une question contenant uniquement des espaces
    retourne une erreur HTTP 400.
    """

    class FakeInvalidRAGSystem:
        def ask(self, question):
            raise ValueError("La question ne peut pas être vide.")

    monkeypatch.setattr(
        main_module,
        "rag_system",
        FakeInvalidRAGSystem(),
    )

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "La question ne peut pas être vide."
    }
    
def test_rebuild_returns_success(monkeypatch):
    """
    Vérifie que /rebuild reconstruit la base vectorielle
    et recharge le système RAG.
    """

    def fake_rebuild_vector_store():
        return {
            "status": "success",
            "events": 100,
            "chunks": 150,
            "vectors": 150,
            "dimension": 1024,
        }

    class FakeReloadedRAGSystem:
        pass

    monkeypatch.setattr(
        main_module,
        "rebuild_vector_store",
        fake_rebuild_vector_store,
    )

    monkeypatch.setattr(
        main_module,
        "RAGSystem",
        FakeReloadedRAGSystem,
    )

    response = client.post("/rebuild")

    assert response.status_code == 200

    assert response.json() == {
        "status": "success",
        "events": 100,
        "chunks": 150,
        "vectors": 150,
        "dimension": 1024,
    }

    assert isinstance(
        main_module.rag_system,
        FakeReloadedRAGSystem,
    )
    
def test_rebuild_handles_error(monkeypatch):
    """
    Vérifie que /rebuild retourne une erreur HTTP 500
    si la reconstruction échoue.
    """

    def fake_rebuild_vector_store():
        raise RuntimeError("Erreur simulée")

    monkeypatch.setattr(
        main_module,
        "rebuild_vector_store",
        fake_rebuild_vector_store,
    )

    response = client.post("/rebuild")

    assert response.status_code == 500
    assert "Erreur lors de la reconstruction" in response.json()["detail"]