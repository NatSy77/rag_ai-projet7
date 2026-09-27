"""
Script de lancement local de l'API RAG.

Usage depuis la racine du projet :
    python scripts/run_api.py
"""

from pathlib import Path

import uvicorn


if __name__ == "__main__":
    # Racine du projet rag_ai-projet7.
    project_root = Path(__file__).resolve().parent.parent

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(project_root),
    )