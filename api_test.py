import requests


API_URL = "http://127.0.0.1:8000"


def test_health():
    """
    Vérifie que l'API est bien démarrée.
    """
    response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )

    print("GET /health")
    print("Status :", response.status_code)
    print("Réponse :", response.json())


def test_ask():
    """
    Envoie une vraie question au système RAG.
    """
    payload = {
        "question": "Quels concerts de jazz sont disponibles à Paris ?"
    }

    response = requests.post(
        f"{API_URL}/ask",
        json=payload,
        timeout=120,
    )

    print("\nPOST /ask")
    print("Status :", response.status_code)

    data = response.json()

    print("\nRéponse générée :")
    print(data.get("answer"))

    print("\nSources :")

    for source in data.get("sources", []):
        print(
            "-",
            source.get("title"),
            "| date :",
            source.get("date"),
            "| score :",
            source.get("similarity_score"),
        )


if __name__ == "__main__":
    test_health()
    test_ask()