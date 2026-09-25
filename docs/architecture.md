# Architecture du système RAG

Le système suit une architecture RAG (Retrieval-Augmented Generation) permettant de rechercher des événements culturels pertinents avant de générer une réponse en langage naturel.

```mermaid
flowchart TD
    U[Utilisateur] --> API[API REST FastAPI]
    API --> RAG[RAGSystem]

    RAG --> EMB[Mistral Embed]
    EMB --> QV[Embedding de la question]
    QV --> FAISS[Index vectoriel FAISS]

    FAISS --> RET[Chunks similaires et scores]
    META[event_chunks.csv - Métadonnées] --> RET

    RET --> EVT[Événements pertinents]
    EVT --> CTX[Construction du contexte]

    CTX --> LC[Chaîne LangChain]
    LC --> LLM[Ministral 3B]

    LLM --> RESP[Réponse et sources]
    RESP --> API
    API --> U
```

## Rôle des composants

- **FastAPI** : expose le système RAG à travers une API REST.
- **RAGSystem** : orchestre la recherche des événements et la génération de la réponse.
- **Mistral Embed** : transforme la question utilisateur en vecteur sémantique.
- **FAISS** : recherche les événements sémantiquement proches de la question.
- **Métadonnées des événements** : fournissent les titres, dates, lieux, descriptions et URL associés aux résultats.
- **LangChain** : orchestre le prompt, le contexte récupéré et le modèle de génération.
- **Ministral 3B** : génère la réponse en langage naturel uniquement à partir du contexte fourni.
- **FastAPI** retourne finalement une réponse JSON contenant la réponse générée et les sources utilisées.