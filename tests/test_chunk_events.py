import pandas as pd

from scripts.chunk_events import chunk_events


def test_chunk_events_keeps_metadata():
    """
    Vérifie que les chunks conservent les métadonnées
    de leur événement d'origine.
    """

    df = pd.DataFrame(
        [
            {
                "uid": "event-1",
                "text_for_embedding": "Concert de jazz à Paris.",
                "title_fr": "Concert de jazz",
                "description_fr": "Un concert de jazz.",
                "daterange_fr": "10 septembre 2026",
                "firstdate_begin": "2026-09-10",
                "lastdate_end": "2026-09-10",
                "location_name": "Salle de concert",
                "location_address": "1 rue de Paris",
                "location_postalcode": "75001",
                "location_city": "Paris",
                "canonicalurl": "https://example.com/event-1",
            }
        ]
    )

    chunks = chunk_events(df)

    assert len(chunks) == 1
    assert chunks.iloc[0]["uid"] == "event-1"
    assert chunks.iloc[0]["chunk_index"] == 0
    assert chunks.iloc[0]["title_fr"] == "Concert de jazz"
    assert chunks.iloc[0]["location_city"] == "Paris"
    assert chunks.iloc[0]["chunk_text"] == "Concert de jazz à Paris."


def test_chunk_events_splits_long_text():
    """
    Vérifie qu'un texte dépassant la taille maximale
    est découpé en plusieurs chunks.
    """

    long_text = "Concert et musique à Paris. " * 100

    df = pd.DataFrame(
        [
            {
                "uid": "event-2",
                "text_for_embedding": long_text,
                "title_fr": "Grand événement musical",
                "description_fr": "Concert",
                "daterange_fr": "Septembre 2026",
                "firstdate_begin": "2026-09-01",
                "lastdate_end": "2026-09-30",
                "location_name": "Paris",
                "location_address": "Paris",
                "location_postalcode": "75001",
                "location_city": "Paris",
                "canonicalurl": "https://example.com/event-2",
            }
        ]
    )

    chunks = chunk_events(df)

    assert len(chunks) > 1

    # Tous les chunks doivent appartenir au même événement.
    assert chunks["uid"].nunique() == 1
    assert chunks["uid"].iloc[0] == "event-2"

    # Les numéros des chunks doivent commencer à zéro
    # et se suivre sans interruption.
    assert chunks["chunk_index"].tolist() == list(
        range(len(chunks))
    )


def test_chunk_events_ignores_empty_text():
    """
    Vérifie qu'un événement sans texte exploitable
    n'est pas ajouté aux chunks.
    """

    df = pd.DataFrame(
        [
            {
                "uid": "event-empty",
                "text_for_embedding": None,
            }
        ]
    )

    chunks = chunk_events(df)

    assert chunks.empty