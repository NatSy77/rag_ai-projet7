import sys
from pathlib import Path
from unittest.mock import patch

# Ajoute le dossier scripts au chemin Python pour pouvoir
# importer la fonction fetch_events depuis les tests.
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "scripts")
)

from fetch_events import fetch_events


@patch("fetch_events.requests.get")
def test_fetch_events_returns_results(mock_get):
    """
    Vérifie que fetch_events() retourne correctement
    les événements fournis par l'API.
    """

    # Simulation d'une réponse de l'API OpenAgenda.
    mock_get.return_value.json.return_value = {
        "total_count": 2,
        "results": [
            {
                "uid": "event-001",
                "title_fr": "Événement test 1",
            },
            {
                "uid": "event-002",
                "title_fr": "Événement test 2",
            },
        ],
    }

    # Simule le bon fonctionnement de raise_for_status().
    mock_get.return_value.raise_for_status.return_value = None

    events = fetch_events()

    # Vérifie que deux événements ont bien été récupérés.
    assert len(events) == 2

    # Vérifie que les UID récupérés correspondent
    # aux données simulées.
    assert events[0]["uid"] == "event-001"
    assert events[1]["uid"] == "event-002"
    
@patch("fetch_events.requests.get")
def test_fetch_events_handles_pagination(mock_get):
    """
    Vérifie que fetch_events() récupère plusieurs pages
    lorsque l'API retourne un lot complet de 100 événements.
    """

    # Première page : 100 événements.
    first_page = {
        "total_count": 102,
        "results": [
            {
                "uid": f"event-{i:03d}",
                "title_fr": f"Événement {i}",
            }
            for i in range(100)
        ],
    }

    # Deuxième et dernière page : 2 événements.
    second_page = {
        "total_count": 102,
        "results": [
            {
                "uid": "event-100",
                "title_fr": "Événement 100",
            },
            {
                "uid": "event-101",
                "title_fr": "Événement 101",
            },
        ],
    }

    # À chaque appel de requests.get(), le mock retourne
    # successivement la première puis la deuxième page.
    mock_get.return_value.json.side_effect = [
        first_page,
        second_page,
    ]

    mock_get.return_value.raise_for_status.return_value = None

    events = fetch_events()

    # Les deux pages doivent avoir été regroupées.
    assert len(events) == 102

    # L'API simulée doit avoir été appelée deux fois.
    assert mock_get.call_count == 2

    # Vérification du premier et du dernier événement.
    assert events[0]["uid"] == "event-000"
    assert events[-1]["uid"] == "event-101"

    # Vérification des offsets utilisés pour la pagination.
    first_call_params = mock_get.call_args_list[0].kwargs["params"]
    second_call_params = mock_get.call_args_list[1].kwargs["params"]

    assert first_call_params["offset"] == 0
    assert second_call_params["offset"] == 100