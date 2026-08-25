import os
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral
from pathlib import Path
import numpy as np
import time
from mistralai.client.errors.sdkerror import SDKError


# Modèle Mistral utilisé pour générer les représentations vectorielles.
EMBEDDING_MODEL = "mistral-embed"

# Nombre de textes envoyés à Mistral dans chaque requête.
BATCH_SIZE = 10


def get_mistral_client():
    """
    Initialise et retourne le client Mistral.

    La clé API est chargée depuis le fichier .env afin
    qu'aucun secret ne soit stocké directement dans le code.
    """
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable MISTRAL_API_KEY est absente. "
            "Vérifiez le fichier .env."
        )

    return Mistral(api_key=api_key)


def generate_embeddings(texts, batch_size=BATCH_SIZE):
    """
    Génère les embeddings Mistral d'une liste de textes.

    Les textes sont envoyés par lots afin d'éviter d'envoyer
    l'ensemble du dataset dans une seule requête.

    Parameters
    ----------
    texts : list[str]
        Textes à transformer en vecteurs.

    batch_size : int
        Nombre de textes envoyés dans chaque requête.

    Returns
    -------
    list[list[float]]
        Liste des vecteurs générés par Mistral.
    """
    client = get_mistral_client()

    all_embeddings = []

    # Parcours des textes par lots.
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch = texts[start:end]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=batch,
        )

        batch_embeddings = [
            item.embedding
            for item in response.data
        ]

        all_embeddings.extend(batch_embeddings)

        print(
            f"Embeddings générés : "
            f"{len(all_embeddings)} / {len(texts)}"
        )

    return all_embeddings

def generate_embeddings_with_checkpoint(
    texts,
    batch_size=BATCH_SIZE,
    embeddings_path="data/embeddings.npy",
    progress_path="data/embedding_progress.txt",
):
    """
    Génère les embeddings par lots avec sauvegarde progressive.

    Après chaque lot traité :
    - les embeddings déjà générés sont sauvegardés ;
    - le nombre de textes traités est enregistré.

    Si le traitement est interrompu, une exécution suivante
    peut reprendre à partir du dernier lot sauvegardé.

    Parameters
    ----------
    texts : list[str]
        Textes à transformer en vecteurs.

    batch_size : int
        Nombre de textes envoyés à Mistral par requête.

    embeddings_path : str
        Chemin du fichier NumPy contenant les embeddings.

    progress_path : str
        Chemin du fichier indiquant le nombre de textes traités.

    Returns
    -------
    numpy.ndarray
        Matrice contenant l'ensemble des embeddings générés.
    """

    client = get_mistral_client()

    embeddings_path = Path(embeddings_path)
    progress_path = Path(progress_path)

    # Création du dossier de destination si nécessaire.
    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Valeurs utilisées lors d'une première exécution.
    start_index = 0
    all_embeddings = []

    # Si des sauvegardes existent déjà, on reprend
    # à partir du dernier point de progression.
    if embeddings_path.exists() and progress_path.exists():
        saved_embeddings = np.load(embeddings_path)

        with open(progress_path, "r", encoding="utf-8") as file:
            start_index = int(file.read().strip())

        # Vérification de cohérence entre les deux sauvegardes.
        if len(saved_embeddings) != start_index:
            raise ValueError(
                "Le fichier d'embeddings et le fichier "
                "de progression sont incohérents."
            )

        all_embeddings = saved_embeddings.tolist()

        print(
            f"Reprise de la vectorisation à partir "
            f"de l'événement {start_index}."
        )

    # Traitement uniquement des textes qui n'ont
    # pas encore été vectorisés.
    for start in range(start_index, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        # Appel à l'API Mistral avec gestion des limites de débit.
        while True:
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=batch,
                )

                # Petite pause pour respecter la limite
                # d'environ une requête par seconde.
                time.sleep(1.1)

                break

            except SDKError as error:
                error_message = str(error)

                if "429" in error_message:
                    wait_time = 5
                    print(
                        f"Limite Mistral atteinte (429). "
                        f"Nouvelle tentative dans {wait_time} secondes..."
                    )

                elif any(
                    code in error_message
                    for code in ["500", "502", "503", "504"]
                ):
                    wait_time = 10
                    print(
                        f"Erreur temporaire Mistral. "
                        f"Nouvelle tentative dans {wait_time} secondes..."
                    )

                else:
                    raise

                time.sleep(wait_time)

        batch_embeddings = [
            item.embedding
            for item in response.data
        ]

        all_embeddings.extend(batch_embeddings)

        # Conversion en float32 : précision suffisante pour
        # FAISS tout en réduisant la taille du fichier.
        embeddings_array = np.asarray(
            all_embeddings,
            dtype=np.float32,
        )

        # Sauvegarde des vecteurs après chaque lot.
        np.save(
            embeddings_path,
            embeddings_array,
        )

        # Sauvegarde de la progression après le même lot.
        with open(
            progress_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(str(end))

        print(
            f"Embeddings sauvegardés : "
            f"{end} / {len(texts)}"
        )

    return np.asarray(
        all_embeddings,
        dtype=np.float32,
    )

if __name__ == "__main__":
    # Chargement du dataset nettoyé et préparé
    # pour la génération des embeddings.
    df = pd.read_csv("data/events_clean.csv")

    # Récupération de l'ensemble des textes à vectoriser.
    texts = df["text_for_embedding"].fillna("").tolist()

    print("Nombre total d'événements à vectoriser :", len(texts))

    # Génération des embeddings avec sauvegarde progressive.
    # En cas d'interruption, une nouvelle exécution reprendra
    # automatiquement à partir du dernier lot sauvegardé.
    embeddings = generate_embeddings_with_checkpoint(
        texts,
        batch_size=50,
        embeddings_path="data/embeddings.npy",
        progress_path="data/embedding_progress.txt",
    )

    print("\nVectorisation terminée.")
    print("Nombre de textes :", len(texts))
    print("Nombre de vecteurs :", len(embeddings))

    if len(embeddings) > 0:
        print(
            "Dimension d'un vecteur :",
            embeddings.shape[1],
        )