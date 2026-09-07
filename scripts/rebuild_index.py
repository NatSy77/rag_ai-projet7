"""
Orchestration complète de la reconstruction de la base vectorielle.

Ce module réutilise les fonctions existantes du projet pour :
- récupérer les événements ;
- les prétraiter ;
- les découper en chunks ;
- générer les embeddings ;
- reconstruire l'index FAISS.

La fonction principale pourra être appelée :
- depuis l'endpoint FastAPI /rebuild ;
- depuis un script local ;
- plus tard depuis Docker.
"""

from pathlib import Path

import faiss
import numpy as np

from scripts.fetch_events import fetch_events
from scripts.preprocess_events import preprocess_events
from scripts.chunk_events import chunk_events
from scripts.generate_embeddings import generate_embeddings_with_checkpoint
from scripts.build_faiss_index import build_faiss_index


DATA_DIR = Path("data")

EVENTS_CLEAN_PATH = DATA_DIR / "events_clean.csv"
CHUNKS_PATH = DATA_DIR / "event_chunks.csv"
EMBEDDINGS_PATH = DATA_DIR / "chunk_embeddings.npy"
PROGRESS_PATH = DATA_DIR / "chunk_embedding_progress.txt"
FAISS_INDEX_PATH = DATA_DIR / "chunks.index"


def rebuild_vector_store():
    """
    Reconstruit entièrement la base vectorielle du projet.

    Returns
    -------
    dict
        Informations résumant la reconstruction.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 - Récupération des événements...")
    events = fetch_events()

    print("2/5 - Prétraitement des événements...")
    clean_df = preprocess_events(events)
    clean_df.to_csv(
        EVENTS_CLEAN_PATH,
        index=False,
    )

    print("3/5 - Création des chunks...")
    chunks_df = chunk_events(clean_df)
    chunks_df.to_csv(
        CHUNKS_PATH,
        index=False,
    )

    # Une reconstruction complète doit repartir de zéro.
    # On supprime donc les anciens fichiers de checkpoint afin
    # d'éviter de réutiliser des embeddings associés à d'anciens chunks.
    for checkpoint_path in [EMBEDDINGS_PATH, PROGRESS_PATH]:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
    print("4/5 - Génération des embeddings...")
    embeddings = generate_embeddings_with_checkpoint(
        chunks_df["chunk_text"].tolist(),
        embeddings_path=str(EMBEDDINGS_PATH),
        progress_path=str(PROGRESS_PATH),
    )

    # Sauvegarde finale explicite.
    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    print("5/5 - Reconstruction de l'index FAISS...")
    index = build_faiss_index(embeddings)

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    return {
        "status": "success",
        "events": len(clean_df),
        "chunks": len(chunks_df),
        "vectors": index.ntotal,
        "dimension": index.d,
    }


if __name__ == "__main__":
    result = rebuild_vector_store()

    print("\nReconstruction terminée :")
    print(result)