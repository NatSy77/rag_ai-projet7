import requests
import pandas as pd


API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)

# Filtres retenus pour le POC
WHERE_FILTER = 'location_city="Paris" AND lastdate_end >= "2025-08-23"'

# Nombre maximal de résultats récupérés par requête
LIMIT = 100


def fetch_events():
    """Récupère les événements OpenAgenda correspondant aux filtres."""

    all_events = []
    offset = 0

    while True:
        params = {
            "where": WHERE_FILTER,
            "limit": LIMIT,
            "offset": offset,
            "order_by": "lastdate_end",
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        events = data["results"]

        all_events.extend(events)

        print(
            f"Événements récupérés : "
            f"{len(all_events)} / {data['total_count']}"
        )

        # Si le dernier lot contient moins de 100 événements,
        # nous avons atteint la fin des résultats.
        if len(events) < LIMIT:
            break

        offset += LIMIT

    return all_events


if __name__ == "__main__":
    events = fetch_events()
    
    df = pd.DataFrame(events)

    print("\nDimensions du DataFrame :", df.shape)
    print("\nColonnes disponibles :")
    print(df.columns.tolist())

    print("\nRécupération terminée.")
    print("Nombre total d'événements :", len(events))
    
    selected_columns = [
    "uid",
    "canonicalurl",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "daterange_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_name",
    "location_address",
    "location_postalcode",
    "location_city",
    "location_department",
    "location_region",
    "location_countrycode",
    "category",
    "age_min",
    "age_max",
    "registration",
]

df = df[selected_columns]

print("\nDimensions après sélection :", df.shape)
print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

missing_title = df["title_fr"].isna()
missing_description = df["description_fr"].isna()

print("\nÉvénements sans titre :", missing_title.sum())
print("Événements sans description :", missing_description.sum())

print(
    "Événements sans titre ET sans description :",
    (missing_title & missing_description).sum()
)

print("\nUID dupliqués :", df["uid"].duplicated().sum())

# Suppression de la colonne category car elle est entièrement vide
df = df.drop(columns=["category"])

# Suppression des événements sans titre et sans description
df = df.dropna(
    subset=["title_fr", "description_fr"],
    how="all"
)

print("\nDimensions après nettoyage :", df.shape)

print("\nValeurs manquantes après nettoyage :")
print(df.isna().sum())

print("\nTypes des colonnes :")
print(df.dtypes)

# Conversion des colonnes de dates
df["firstdate_begin"] = pd.to_datetime(
    df["firstdate_begin"],
    utc=True,
    errors="coerce"
)

df["lastdate_end"] = pd.to_datetime(
    df["lastdate_end"],
    utc=True,
    errors="coerce"
)

print("\nDates invalides après conversion :")
print("firstdate_begin :", df["firstdate_begin"].isna().sum())
print("lastdate_end    :", df["lastdate_end"].isna().sum())

print("\nTypes des dates :")
print(df[["firstdate_begin", "lastdate_end"]].dtypes)

print("\nExemples de keywords_fr :")
print(df["keywords_fr"].dropna().head())

print("\nTypes rencontrés dans keywords_fr :")
print(df["keywords_fr"].dropna().map(type).value_counts())

print("\nExemples de registration :")
print(df["registration"].dropna().head())

print("\nVilles présentes :")
print(df["location_city"].value_counts())

print("\nCodes pays présents :")
print(df["location_countrycode"].value_counts())

print("\nRégions présentes :")
print(df["location_region"].value_counts(dropna=False))

# Normalisation du code pays
df["location_countrycode"] = (
    df["location_countrycode"]
    .str.upper()
)

# Normalisation de la région pour les événements localisés à Paris
df["location_region"] = "Île-de-France"

print("\nAprès normalisation géographique :")

print("\nCodes pays :")
print(df["location_countrycode"].value_counts(dropna=False))

print("\nRégions :")
print(df["location_region"].value_counts(dropna=False))

print("\nCodes postaux les plus fréquents :")
print(df["location_postalcode"].value_counts(dropna=False).head(30))

print("\nNombre de codes postaux différents :")
print(df["location_postalcode"].nunique())

valid_paris_postal_codes = [
    f"750{i:02d}" for i in range(1, 21)
] + ["75116"]

invalid_postal_codes = df[
    df["location_postalcode"].notna()
    & ~df["location_postalcode"].isin(valid_paris_postal_codes)
]

print("\nÉvénements avec un code postal inhabituel :")
print(
    invalid_postal_codes[
        [
            "title_fr",
            "location_name",
            "location_address",
            "location_postalcode",
            "location_city",
        ]
    ].to_string(index=False)
)

print(
    "\nNombre d'événements avec un code postal inhabituel :",
    len(invalid_postal_codes)
)