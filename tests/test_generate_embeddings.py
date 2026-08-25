import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Permet d'importer les scripts du projet depuis les tests.
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "scripts")
)

from generate_embeddings import generate_embeddings


@patch("generate_embeddings.get_mistral_client")
def test_generate_embeddings_uses_batches(mock_get_client):
    """
    Vérifie que generate_embeddings() découpe correctement
    les textes en plusieurs lots et regroupe tous les vecteurs.
    """

    # Création d'un faux client Mistral.
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # 25 textes fictifs.
    texts = [
        f"Texte {i}"
        for i in range(25)
    ]

    # Simulation de trois réponses Mistral :
    # 10 vecteurs, puis 10, puis 5.
    responses = []

    for batch_size in [10, 10, 5]:
        response = MagicMock()

        response.data = [
            MagicMock(embedding=[float(i), 0.0])
            for i in range(batch_size)
        ]

        responses.append(response)

    mock_client.embeddings.create.side_effect = responses

    embeddings = generate_embeddings(
        texts,
        batch_size=10,
    )

    # 25 textes doivent produire 25 vecteurs.
    assert len(embeddings) == 25

    # Trois lots doivent provoquer trois appels à l'API simulée.
    assert mock_client.embeddings.create.call_count == 3

@patch("generate_embeddings.get_mistral_client")
def test_generate_embeddings_preserves_order(mock_get_client):
    """
    Vérifie que l'ordre des embeddings retournés correspond
    à l'ordre des textes fournis, même avec plusieurs lots.
    """

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    texts = [
        "Texte 0",
        "Texte 1",
        "Texte 2",
        "Texte 3",
    ]

    # Premier lot : textes 0 et 1.
    first_response = MagicMock()
    first_response.data = [
        MagicMock(embedding=[0.0, 0.0]),
        MagicMock(embedding=[1.0, 0.0]),
    ]

    # Deuxième lot : textes 2 et 3.
    second_response = MagicMock()
    second_response.data = [
        MagicMock(embedding=[2.0, 0.0]),
        MagicMock(embedding=[3.0, 0.0]),
    ]

    mock_client.embeddings.create.side_effect = [
        first_response,
        second_response,
    ]

    embeddings = generate_embeddings(
        texts,
        batch_size=2,
    )

    assert embeddings == [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]