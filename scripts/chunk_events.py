from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Dataset nettoyé contenant les textes préparés pour les embeddings.
DATA_PATH = Path("data/events_clean.csv")

# Paramètres de découpage choisis après analyse
# de la longueur des textes du dataset.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_events(df):
    """
    Découpe le texte de chaque événement en plusieurs chunks.

    Chaque chunk conserve l'UID de l'événement d'origine ainsi
    que son numéro afin de pouvoir ensuite associer les vecteurs
    FAISS aux métadonnées de l'événement.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset des événements nettoyés.

    Returns
    -------
    pandas.DataFrame
        DataFrame contenant un chunk par ligne.
    """

    # RecursiveCharacterTextSplitter essaie de conserver
    # autant que possible les frontières naturelles du texte.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = []

    # Découpage événement par événement.
    for _, event in df.iterrows():
        text = event["text_for_embedding"]

        if pd.isna(text) or not str(text).strip():
            continue

        event_chunks = text_splitter.split_text(str(text))
        
        # Si le dernier chunk est très court, on le fusionne
        # avec le précédent afin de conserver suffisamment
        # de contexte sémantique.
        if (
            len(event_chunks) > 1
            and len(event_chunks[-1]) < 200
        ):
            event_chunks[-2] = (
                event_chunks[-2]
                + "\n"
                + event_chunks[-1]
            )

            event_chunks.pop()

        # Chaque chunk garde une référence vers
        # l'événement dont il provient.
        for chunk_index, chunk_text in enumerate(event_chunks):
            # Chaque chunk conserve les métadonnées principales
            # de son événement d'origine. Elles permettront de
            # reconstruire un résultat compréhensible après une
            # recherche dans l'index FAISS.
            chunks.append(
                {
                    "uid": event["uid"],
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "title_fr": event["title_fr"],
                    "description_fr": event["description_fr"],
                    "daterange_fr": event["daterange_fr"],
                    "firstdate_begin": event["firstdate_begin"],
                    "lastdate_end": event["lastdate_end"],
                    "location_name": event["location_name"],
                    "location_address": event["location_address"],
                    "location_postalcode": event[
                        "location_postalcode"
                    ],
                    "location_city": event["location_city"],
                    "canonicalurl": event["canonicalurl"],
                }
            )

    return pd.DataFrame(chunks)


if __name__ == "__main__":
    # Chargement des événements préparés lors de l'étape 2.
    df = pd.read_csv(DATA_PATH)

    print("Nombre d'événements :", len(df))

    # Découpage des textes.
    chunks_df = chunk_events(df)

    print("Nombre total de chunks :", len(chunks_df))

    # Statistiques permettant de vérifier la stratégie choisie.
    chunks_per_event = chunks_df.groupby("uid").size()

    print(
        "Nombre moyen de chunks par événement :",
        round(chunks_per_event.mean(), 2),
    )

    print(
        "Nombre maximum de chunks pour un événement :",
        chunks_per_event.max(),
    )

    print("\nDistribution du nombre de chunks par événement :")
    print(
        chunks_per_event.value_counts()
        .sort_index()
        .head(15)
    )

    print("\nLongueur des chunks :")
    print(chunks_df["chunk_text"].str.len().describe())
    
    chunk_lengths = chunks_df["chunk_text"].str.len()

    print("\nChunks très courts :")
    print("Moins de 100 caractères :", (chunk_lengths < 100).sum())
    print("Moins de 200 caractères :", (chunk_lengths < 200).sum())
    
    # Sauvegarde des chunks et de leurs métadonnées.
    # Ce fichier servira ensuite de correspondance entre
    # les positions de l'index FAISS et les événements.
    output_path = Path("data/event_chunks.csv")

    chunks_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nChunks et métadonnées sauvegardés dans :",
        output_path,
    )

    print(
        "Dimensions du fichier :",
        chunks_df.shape,
    )