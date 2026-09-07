"""
Schémas Pydantic utilisés par l'API FastAPI.

Ils définissent le format des données reçues et retournées
par les endpoints de l'application.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Question envoyée au système RAG."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question utilisateur sur les événements culturels.",
        examples=["Quels concerts de jazz sont disponibles à Paris ?"],
    )


class Source(BaseModel):
    """Source utilisée par le système RAG pour générer sa réponse."""

    title: str
    date: str | None = None
    location: str | None = None
    url: str | None = None
    similarity_score: float


class AskResponse(BaseModel):
    """Réponse retournée par le système RAG."""

    answer: str
    context: str
    sources: list[Source]
    
class RebuildResponse(BaseModel):
    """Résultat de la reconstruction de la base vectorielle."""

    status: str
    events: int
    chunks: int
    vectors: int
    dimension: int