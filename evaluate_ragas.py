import os

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy

from scripts.rag_system import RAGSystem


QUESTIONS_PATH = "evaluation_questions.csv"
RESULTS_PATH = "evaluation_ragas_results.csv"


# ---------------------------------------------------------
# 1. Charger la clé API Mistral
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError(
        "MISTRAL_API_KEY est absente du fichier .env."
    )


# ---------------------------------------------------------
# 2. Initialiser le client compatible OpenAI
# ---------------------------------------------------------
#
# L'API Mistral propose une interface compatible OpenAI.
# Cela permet à Ragas d'utiliser les modèles Mistral.
#

async_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)


# ---------------------------------------------------------
# 3. Initialiser le LLM juge et les embeddings
# ---------------------------------------------------------

evaluator_llm = llm_factory(
    model="ministral-3b-2512",
    provider="openai",
    client=async_client,
)

evaluator_embeddings = embedding_factory(
    provider="openai",
    model="mistral-embed",
    client=async_client,
)


# ---------------------------------------------------------
# 4. Créer les métriques Ragas
# ---------------------------------------------------------

faithfulness_metric = Faithfulness(
    llm=evaluator_llm,
)

relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)


# ---------------------------------------------------------
# 5. Charger le jeu d'évaluation
# ---------------------------------------------------------

questions_df = pd.read_csv(QUESTIONS_PATH)

# Une nouvelle exécution produit une nouvelle évaluation
# complète des questions.
results = []

# Initialiser le système RAG une seule fois.
rag = RAGSystem()


# ---------------------------------------------------------
# 6. Évaluer toutes les questions
# ---------------------------------------------------------

for index, row in questions_df.iterrows():

    question = row["question"]
    category = row["category"]

    print(
        f"\nQuestion {index + 1}/{len(questions_df)} : "
        f"{question}"
    )
    print(f"Catégorie : {category}")

    # Initialiser le score à None pour éviter de réutiliser
    # accidentellement celui de la question précédente.
    mean_similarity_score = None

    try:
        # -------------------------------------------------
        # Exécuter le système RAG
        # -------------------------------------------------

        rag_result = rag.ask(question)

        response = rag_result["answer"]

        # Contexte réellement utilisé par le RAG
        # pour générer sa réponse.
        contexts = [rag_result["context"]]

        # -------------------------------------------------
        # Calculer le score de similarité FAISS moyen
        # -------------------------------------------------

        similarity_scores = [
            source["similarity_score"]
            for source in rag_result["sources"]
        ]

        if similarity_scores:
            mean_similarity_score = (
                sum(similarity_scores)
                / len(similarity_scores)
            )

        # -------------------------------------------------
        # Évaluer la réponse avec Ragas
        # -------------------------------------------------

        print("Évaluation Ragas en cours...")

        faithfulness_result = faithfulness_metric.score(
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
        )

        relevancy_result = relevancy_metric.score(
            user_input=question,
            response=response,
        )

        print(
            "Faithfulness :",
            faithfulness_result,
        )
        print(
            "Answer Relevancy :",
            relevancy_result,
        )
        print(
            "FAISS Similarity Score moyen :",
            (
                round(mean_similarity_score, 4)
                if mean_similarity_score is not None
                else "N/A"
            ),
        )

        # -------------------------------------------------
        # Enregistrer les résultats
        # -------------------------------------------------

        results.append(
            {
                "question": question,
                "category": category,
                "faithfulness": faithfulness_result.value,
                "answer_relevancy": relevancy_result.value,
                "mean_similarity_score": mean_similarity_score,
                "error": "",
            }
        )

    except Exception as exc:
        # Une erreur d'évaluation ne doit pas interrompre
        # l'ensemble du jeu de tests.
        #
        # Si FAISS a déjà fonctionné avant l'erreur Ragas,
        # son score est conservé.

        print(
            "Erreur pendant l'évaluation :",
            exc,
        )

        results.append(
            {
                "question": question,
                "category": category,
                "faithfulness": None,
                "answer_relevancy": None,
                "mean_similarity_score": mean_similarity_score,
                "error": str(exc),
            }
        )

    # -----------------------------------------------------
    # Sauvegarde progressive
    # -----------------------------------------------------
    #
    # Le CSV est sauvegardé après chaque question afin
    # de conserver les résultats déjà obtenus en cas
    # d'erreur ultérieure.

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        f"Progression sauvegardée : "
        f"{index + 1}/{len(questions_df)}"
    )


# ---------------------------------------------------------
# 7. Résumé final
# ---------------------------------------------------------

print("\nÉvaluation terminée.")
print(
    f"Résultats sauvegardés dans {RESULTS_PATH}"
)