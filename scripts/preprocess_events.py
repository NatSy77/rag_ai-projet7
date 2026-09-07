import pandas as pd
from scripts.fetch_events import fetch_events
import re
from html import unescape

# Colonnes conservées pour le futur système RAG.
# Elles regroupent les informations nécessaires pour répondre
# aux principales questions sur un événement :
# quoi, quand, où, pour quel âge et comment s'inscrire.
SELECTED_COLUMNS = [
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

def clean_html(text):
    """
    Nettoie une chaîne contenant du HTML.

    Les balises HTML sont supprimées et les entités HTML
    sont converties en caractères lisibles.
    """
    if pd.isna(text):
        return None

    # Conversion des entités HTML (&amp;, &quot;, etc.).
    text = unescape(str(text))

    # Remplacement de certaines balises de séparation
    # par des espaces avant suppression des autres balises.
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", " ", text, flags=re.IGNORECASE)

    # Suppression des balises HTML restantes.
    text = re.sub(r"<[^>]+>", "", text)

    # Suppression des espaces multiples.
    text = re.sub(r"\s+", " ", text).strip()

    return text

def build_text_for_embedding(row):
    """
    Construit le texte qui sera utilisé pour générer
    l'embedding d'un événement.

    Seules les informations disponibles sont ajoutées
    afin d'éviter d'inclure des valeurs manquantes.
    """

    parts = []

    # Titre de l'événement
    if pd.notna(row["title_fr"]):
        parts.append(f"Titre : {row['title_fr']}")

    # Description courte
    if pd.notna(row["description_fr"]):
        parts.append(f"Description : {row['description_fr']}")
        
    if pd.notna(row["longdescription_fr"]):
        long_description = clean_html(
            row["longdescription_fr"]
        )

        if long_description:
            parts.append(
                f"Description détaillée : {long_description}"
            )

    # Mots-clés OpenAgenda
    if isinstance(row["keywords_fr"], list):
        keywords = [
            str(keyword).strip()
            for keyword in row["keywords_fr"]
            if str(keyword).strip()
        ]

        if keywords:
            parts.append(
                "Mots-clés : " + ", ".join(keywords)
            )

    # Informations temporelles
    if pd.notna(row["daterange_fr"]):
        parts.append(f"Dates : {row['daterange_fr']}")
        
    if pd.notna(row["firstdate_begin"]):
        parts.append(
            "Début : "
            + row["firstdate_begin"].strftime("%d/%m/%Y %H:%M")
        )

    if pd.notna(row["lastdate_end"]):
        parts.append(
            "Fin : "
            + row["lastdate_end"].strftime("%d/%m/%Y %H:%M")
        )

    # Informations sur le lieu
    location_parts = []

    if pd.notna(row["location_name"]):
        location_parts.append(str(row["location_name"]))

    if pd.notna(row["location_address"]):
        location_parts.append(str(row["location_address"]))

    if pd.notna(row["location_city"]):
        location_parts.append(str(row["location_city"]))

    if location_parts:
        parts.append(
            "Lieu : " + ", ".join(location_parts)
        )

    # Restrictions d'âge lorsqu'elles sont renseignées
    if pd.notna(row["age_min"]):
        parts.append(
            f"Âge minimum : {int(row['age_min'])} ans"
        )

    if pd.notna(row["age_max"]):
        parts.append(
            f"Âge maximum : {int(row['age_max'])} ans"
        )

    return "\n".join(parts)

def preprocess_events(events):
    """
    Nettoie et structure les événements récupérés depuis OpenAgenda.

    Les principales opérations réalisées sont :
    - conversion des données en DataFrame Pandas ;
    - sélection des colonnes utiles au projet ;
    - suppression de la colonne 'category' entièrement vide ;
    - suppression des événements sans titre ni description ;
    - conversion des dates au format datetime ;
    - normalisation des informations géographiques.

    Parameters
    ----------
    events : list
        Liste des événements bruts récupérés depuis OpenAgenda.

    Returns
    -------
    pandas.DataFrame
        DataFrame nettoyé et structuré, prêt pour les étapes suivantes.
    """

    # Transformation de la liste de dictionnaires en DataFrame.
    df = pd.DataFrame(events)

    print("\nDimensions avant pré-processing :", df.shape)

    # Conservation uniquement des colonnes utiles au POC.
    df = df[SELECTED_COLUMNS].copy()

    # La colonne 'category' est vide pour tous les événements
    # récupérés dans notre périmètre. Elle n'apporte donc
    # aucune information exploitable au système RAG.
    df = df.drop(columns=["category"])

    # Les événements sans titre ET sans description ne possèdent
    # pas suffisamment de contenu textuel pour être exploités
    # correctement par le futur système RAG.
    df = df.dropna(
        subset=["title_fr", "description_fr"],
        how="all",
    )

    # Conversion des dates fournies sous forme de chaînes de caractères
    # en véritables objets datetime Pandas.
    #
    # utc=True permet d'uniformiser le fuseau horaire.
    # errors="coerce" transforme une éventuelle date invalide en NaT
    # plutôt que de provoquer l'arrêt du programme.
    df["firstdate_begin"] = pd.to_datetime(
        df["firstdate_begin"],
        utc=True,
        errors="coerce",
    )

    df["lastdate_end"] = pd.to_datetime(
        df["lastdate_end"],
        utc=True,
        errors="coerce",
    )

    # Uniformisation du code pays.
    # Certaines lignes contiennent "FR" et d'autres "fr".
    df["location_countrycode"] = (
        df["location_countrycode"]
        .str.upper()
    )

    # Tous les événements sélectionnés ont location_city="Paris".
    # Plusieurs variantes ou anomalies existent cependant dans
    # location_region : "Île-de-France", "IDF", "Paris", etc.
    # La région est donc normalisée.
    df["location_region"] = "Île-de-France"
    
    # Construction du contenu textuel qui sera utilisé
    # lors de la future génération des embeddings.
    df["text_for_embedding"] = df.apply(
        build_text_for_embedding,
        axis=1,
    )

    # Réinitialisation de l'index après suppression de lignes.
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    # 1. Récupération des données depuis OpenAgenda.
    events = fetch_events()

    # 2. Nettoyage et structuration.
    df_clean = preprocess_events(events)

    # Quelques contrôles permettant de vérifier le résultat
    # du pré-processing.
    print("\nPré-processing terminé.")
    print("Dimensions finales :", df_clean.shape)

    print(
        "UID dupliqués :",
        df_clean["uid"].duplicated().sum(),
    )

    print(
        "Titres manquants :",
        df_clean["title_fr"].isna().sum(),
    )

    print(
        "Descriptions manquantes :",
        df_clean["description_fr"].isna().sum(),
    )

    print(
        "Dates de début invalides :",
        df_clean["firstdate_begin"].isna().sum(),
    )

    print(
        "Dates de fin invalides :",
        df_clean["lastdate_end"].isna().sum(),
    )
    
    print("\nExemple de texte préparé pour l'embedding :\n")
    print(df_clean.iloc[0]["text_for_embedding"])
    
    # Sauvegarde du jeu de données nettoyé.
    # Ce fichier servira de base aux prochaines étapes du projet,
    # notamment à la préparation des données pour la vectorisation.
    output_path = "data/events_clean.csv"

    df_clean.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\nDataset nettoyé sauvegardé dans : {output_path}")