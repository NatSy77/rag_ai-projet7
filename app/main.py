"""
API REST du système de recommandation d'événements culturels.

Cette API expose le système RAG à travers des endpoints HTTP.
"""

from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse, RebuildResponse
from scripts.rag_system import RAGSystem
from fastapi import FastAPI, HTTPException
from scripts.rebuild_index import rebuild_vector_store

app = FastAPI(
    title="API RAG - Événements culturels",
    description=(
        "API permettant d'interroger un système RAG "
        "de recommandation d'événements culturels à Paris."
    ),
    version="1.0.0",
)

# Le système RAG est initialisé une seule fois au démarrage de l'API.
rag_system = RAGSystem()

@app.get("/")
def root():
    """
    Vérifie que l'API est accessible.
    """

    return {
        "message": "API RAG événements culturels opérationnelle"
    }


@app.get("/health")
def health():
    """
    Vérifie l'état de fonctionnement de l'API.
    """

    return {
        "status": "ok"
    }
    
@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question au système RAG",
)
def ask_question(request: AskRequest):
    """
    Envoie une question au système RAG et retourne
    une réponse augmentée avec son contexte et ses sources.
    """

    try:
        result = rag_system.ask(request.question)
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
        
@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire la base vectorielle",
)
def rebuild():
    """
    Reconstruit les données vectorielles et recharge
    le système RAG avec le nouvel index FAISS.

    Attention : cette opération peut être longue car elle
    récupère les événements et régénère les embeddings.
    """

    global rag_system

    try:
        # Reconstruction des fichiers :
        # événements, chunks, embeddings et index FAISS.
        result = rebuild_vector_store()

        # Recharge le RAG afin que /ask utilise immédiatement
        # le nouvel index et les nouvelles métadonnées.
        rag_system = RAGSystem()

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la reconstruction : {exc}",
        ) from exc