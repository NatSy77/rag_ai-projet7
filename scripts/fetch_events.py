import requests


# URL de l'API OpenDataSoft donnant accès aux événements OpenAgenda.
API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)

# Périmètre retenu pour le POC :
# - événements localisés à Paris ;
# - événements dont la dernière occurrence se termine
#   au plus tôt le 23 août 2025.
#
# Cela permet de conserver un an d'historique ainsi
# que les événements en cours et futurs.
WHERE_FILTER = 'location_city="Paris" AND lastdate_end >= "2025-08-23"'

# L'API OpenDataSoft autorise jusqu'à 100 résultats par requête.
LIMIT = 100


def fetch_events():
    """
    Récupère les événements OpenAgenda correspondant au périmètre du POC.

    Les données sont récupérées par pagination, par lots de 100 événements,
    jusqu'à ce que tous les résultats disponibles aient été téléchargés.

    Returns
    -------
    list
        Liste de dictionnaires représentant les événements récupérés.
    """

    all_events = []
    offset = 0

    while True:
        # Paramètres envoyés à l'API pour le lot courant.
        params = {
            "where": WHERE_FILTER,
            "limit": LIMIT,
            "offset": offset,
            "order_by": "lastdate_end",
        }

        # Appel de l'API OpenAgenda.
        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
        )

        # Déclenche une exception HTTP en cas d'erreur
        # (404, 500, etc.).
        response.raise_for_status()

        # Conversion de la réponse JSON en dictionnaire Python.
        data = response.json()

        # Événements présents dans le lot courant.
        events = data["results"]

        # Ajout du lot à la liste globale.
        all_events.extend(events)

        print(
            f"Événements récupérés : "
            f"{len(all_events)} / {data['total_count']}"
        )

        # Si le lot contient moins de LIMIT événements,
        # cela signifie que nous avons atteint la dernière page.
        if len(events) < LIMIT:
            break

        # Passage au lot suivant.
        offset += LIMIT

    return all_events


if __name__ == "__main__":
    events = fetch_events()

    print("\nRécupération terminée.")
    print("Nombre total d'événements :", len(events))