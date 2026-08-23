import sys
from pathlib import Path

import pandas as pd

# Ajoute le dossier scripts au chemin Python afin de pouvoir
# importer la fonction de pré-processing depuis les tests.
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "scripts")
)

from preprocess_events import preprocess_events


def create_valid_event():
    """
    Crée un événement fictif valide utilisé comme base
    pour les différents tests unitaires.
    """
    return {
        "uid": "test-001",
        "canonicalurl": "https://example.com/event",
        "title_fr": "Événement test",
        "description_fr": "Description de l'événement test",
        "longdescription_fr": "Description détaillée",
        "keywords_fr": ["test", "culture"],
        "daterange_fr": "1 septembre 2026",
        "firstdate_begin": "2026-09-01T10:00:00+00:00",
        "lastdate_end": "2026-09-01T12:00:00+00:00",
        "location_name": "Lieu test",
        "location_address": "1 rue de Test",
        "location_postalcode": "75001",
        "location_city": "Paris",
        "location_department": "Paris",
        "location_region": "Ile-de-France",
        "location_countrycode": "fr",
        "category": None,
        "age_min": None,
        "age_max": None,
        "registration": None,
    }


def test_remove_event_without_title_and_description():
    """
    Vérifie qu'un événement sans titre ET sans description
    est supprimé pendant le pré-processing.
    """
    event = create_valid_event()

    event["title_fr"] = None
    event["description_fr"] = None

    df_clean = preprocess_events([event])

    assert len(df_clean) == 0


def test_valid_event_is_kept():
    """
    Vérifie qu'un événement valide est conservé
    après le pré-processing.
    """
    event = create_valid_event()

    df_clean = preprocess_events([event])

    assert len(df_clean) == 1
    assert df_clean.iloc[0]["uid"] == "test-001"


def test_category_column_is_removed():
    """
    Vérifie que la colonne category, inutilisable dans
    notre dataset, est supprimée pendant le pré-processing.
    """
    event = create_valid_event()

    df_clean = preprocess_events([event])

    assert "category" not in df_clean.columns


def test_country_code_is_normalized():
    """
    Vérifie que le code pays est normalisé en majuscules.
    """
    event = create_valid_event()
    event["location_countrycode"] = "fr"

    df_clean = preprocess_events([event])

    assert df_clean.iloc[0]["location_countrycode"] == "FR"


def test_region_is_normalized():
    """
    Vérifie que la région est normalisée en Île-de-France
    pour les événements du périmètre parisien.
    """
    event = create_valid_event()
    event["location_region"] = "IDF"

    df_clean = preprocess_events([event])

    assert df_clean.iloc[0]["location_region"] == "Île-de-France"


def test_dates_are_converted_to_datetime():
    """
    Vérifie que les dates OpenAgenda sont converties
    en véritables dates Pandas.
    """
    event = create_valid_event()

    df_clean = preprocess_events([event])

    assert pd.api.types.is_datetime64_any_dtype(
        df_clean["firstdate_begin"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        df_clean["lastdate_end"]
    )