import os
from pathlib import Path

import faiss
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# Chemins vers les artefacts produits à l'étape 3.
CHUNKS_PATH = Path("data/event_chunks.csv")
FAISS_INDEX_PATH = Path("data/chunks.index")

# Modèles Mistral utilisés dans le système.
EMBEDDING_MODEL = "mistral-embed"
GENERATION_MODEL = "mistral-small-2603"

# Nombre d'événements utilisés comme contexte pour le LLM.
DEFAULT_TOP_K = 5


class RAGSystem:
    """
    Système RAG central du projet.

    Cette classe :
    - charge l'index FAISS et les métadonnées une seule fois ;
    - recherche les événements pertinents ;
    - construit le contexte fourni au LLM ;
    - génère une réponse avec Mistral.

    Elle pourra ensuite être réutilisée directement
    par l'API FastAPI.
    """

    def __init__(
        self,
        chunks_path=CHUNKS_PATH,
        index_path=FAISS_INDEX_PATH,
        client=None,
        chunks_df=None,
        index=None,
        generation_chain=None,
    ):
        """
        Initialise le système RAG.

        Les dépendances peuvent être fournies directement
        pendant les tests unitaires. En utilisation normale,
        elles sont chargées automatiquement.
        """

        # Initialisation du client Mistral uniquement
        # s'il n'est pas fourni explicitement.
        if client is None:
            load_dotenv()

            api_key = os.getenv("MISTRAL_API_KEY")

            if not api_key:
                raise ValueError(
                    "La variable MISTRAL_API_KEY est absente. "
                    "Vérifiez le fichier .env."
                )

            client = Mistral(api_key=api_key)

        self.client = client

        # Chargement des chunks uniquement si aucun DataFrame
        # n'est fourni directement.
        if chunks_df is None:
            chunks_df = pd.read_csv(chunks_path)

        # Initialisation de la chaîne LangChain uniquement
        # si elle n'est pas fournie directement.
        if generation_chain is None:
            self.llm = ChatMistralAI(
                model=GENERATION_MODEL,
                api_key=os.getenv("MISTRAL_API_KEY"),
                temperature=0.2,
                max_tokens=600,
            )

            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "Tu es un assistant spécialisé dans la recommandation "
                            "d'événements culturels à Paris. "
                            "Réponds uniquement à partir des événements présents "
                            "dans le contexte fourni. "
                            "N'invente jamais d'événement, de date, de lieu ou "
                            "d'information absente du contexte. "
                            "Si le contexte ne permet pas de répondre correctement, "
                            "indique clairement que tu ne disposes pas "
                            "d'information suffisante. "
                            "Réponds en français de manière claire et concise."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Question utilisateur :\n{question}\n\n"
                            "Événements disponibles :\n{context}\n\n"
                            "Propose les événements les plus pertinents et indique "
                            "leur date et leur lieu lorsqu'ils sont disponibles."
                        ),
                    ),
                ]
            )

            generation_chain = (
                self.prompt
                | self.llm
                | StrOutputParser()
            )

        self.generation_chain = generation_chain

        self.chunks_df = chunks_df

        # Chargement de l'index FAISS uniquement si aucun index
        # n'est fourni directement.
        if index is None:
            index = faiss.read_index(
                str(index_path)
            )

        self.index = index

        # Vérification de cohérence entre les métadonnées
        # et les vecteurs indexés.
        if len(self.chunks_df) != self.index.ntotal:
            raise ValueError(
                "Le nombre de chunks ne correspond pas "
                "au nombre de vecteurs présents dans FAISS."
            )

    def retrieve(self, question, top_k=DEFAULT_TOP_K):
        """
        Recherche les événements les plus pertinents
        pour une question utilisateur.
        """

        if not question or not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        # Génération de l'embedding de la question.
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[question],
        )

        query_embedding = response.data[0].embedding

        # FAISS attend une matrice float32 à deux dimensions.
        import numpy as np

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        faiss.normalize_L2(query_vector)

        # On récupère davantage de chunks que le nombre final
        # d'événements souhaités afin de pouvoir dédupliquer.
        search_k = min(
            top_k * 20,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_vector,
            search_k,
        )

        results = self.chunks_df.iloc[
            indices[0]
        ].copy()

        results["similarity_score"] = scores[0]
        
        # Conversion de la date de fin pour permettre
        # le filtrage des événements terminés.
        results["lastdate_end"] = pd.to_datetime(
            results["lastdate_end"],
            errors="coerce",
            utc=True,
        )

        # Date actuelle en UTC.
        now = pd.Timestamp.now(tz="UTC")

        # Pour la recommandation, on conserve uniquement
        # les événements encore en cours ou à venir.
        results = results[
            results["lastdate_end"].notna()
            & (results["lastdate_end"] >= now)
        ]

        # Suppression des doublons métier.
        results = (
            results
            .sort_values(
                by="similarity_score",
                ascending=False,
            )
            .drop_duplicates(
                subset=[
                    "title_fr",
                    "daterange_fr",
                    "location_name",
                ],
                keep="first",
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        return results

    def build_context(self, results):
        """
        Construit le contexte textuel transmis au LLM
        à partir des événements retrouvés par FAISS.
        """

        context_parts = []

        for i, (_, event) in enumerate(
            results.iterrows(),
            start=1,
        ):
            context_parts.append(
                f"Événement {i}\n"
                f"Titre : {event['title_fr']}\n"
                f"Date : {event['daterange_fr']}\n"
                f"Lieu : {event['location_name']}\n"
                f"Contenu pertinent : {event['chunk_text']}\n"
                f"URL : {event['canonicalurl']}"
            )

        return "\n\n".join(context_parts)

    def generate_answer(self, question, context):
        """
        Génère une réponse naturelle avec une chaîne LangChain.

        LangChain orchestre ici :
        - le prompt ;
        - l'appel au modèle Mistral ;
        - la récupération du texte généré.
        """

        return self.generation_chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

    def ask(self, question, top_k=DEFAULT_TOP_K):
        """
        Exécute la chaîne RAG complète.

        Returns
        -------
        dict
            Réponse générée, contexte utilisé et sources.
        """

        results = self.retrieve(
            question,
            top_k=top_k,
        )

        context = self.build_context(results)

        answer = self.generate_answer(
            question,
            context,
        )

        # Préparation de sources structurées qui seront
        # directement réutilisables par la future API.
        sources = []

        for _, event in results.iterrows():
            sources.append(
                {
                    "title": event["title_fr"],
                    "date": event["daterange_fr"],
                    "location": event["location_name"],
                    "url": event["canonicalurl"],
                    "similarity_score": float(
                        event["similarity_score"]
                    ),
                }
            )

        return {
            "answer": answer,
            "context": context,
            "sources": sources,
        }


if __name__ == "__main__":
    # Premier test manuel du système RAG complet.
    rag = RAGSystem()

    question = (
        "Peux-tu me recommander un bon restaurant japonais à Lyon ?"
    )

    result = rag.ask(question)

    print("\nQuestion :")
    print(question)

    print("\nRéponse :")
    print(result["answer"])

    print("\nSources utilisées :")
    for source in result["sources"]:
        print(
            f"- {source['title']} "
            f"({source['similarity_score']:.4f})"
        )