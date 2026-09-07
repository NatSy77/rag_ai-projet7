import os

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy


# ---------------------------------------------------------
# 1. Charger la clé API Mistral
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY est absente du fichier .env.")


# ---------------------------------------------------------
# 2. Initialiser les clients compatibles OpenAI
# ---------------------------------------------------------
#
# L'API Mistral propose une interface compatible OpenAI.
# Cela permet à Ragas de fonctionner avec les modèles Mistral
# malgré une incompatibilité rencontrée avec son adaptateur natif.
#

async_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)

sync_client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1",
)


# ---------------------------------------------------------
# 3. Initialiser le LLM juge et les embeddings
# ---------------------------------------------------------

evaluator_llm = llm_factory(
    model="mistral-small-2603",
    provider="openai",
    client=async_client,
)

evaluator_embeddings = embedding_factory(
    provider="openai",
    model="mistral-embed",
    client=sync_client,
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
# 5. Exemple représentatif issu du système RAG
# ---------------------------------------------------------

question = "Quels concerts de jazz sont disponibles à Paris ?"

response = (
    "Plusieurs concerts de jazz sont proposés à Paris, "
    "notamment le Chet Baker Tribute et le Gaël Horellou Organ Trio."
)

contexts = [
    (
        "Chet Baker Tribute / Riccardo Del Fra Quartet. "
        "Concert de jazz à Paris le 17 septembre."
    ),
    (
        "Gaël Horellou Organ Trio. "
        "Concert de jazz à Paris le 16 septembre."
    ),
]


# ---------------------------------------------------------
# 6. Lancer l'évaluation
# ---------------------------------------------------------

print("Évaluation Ragas en cours...")

try:
    faithfulness_result = faithfulness_metric.score(
        user_input=question,
        response=response,
        retrieved_contexts=contexts,
    )

    relevancy_result = relevancy_metric.score(
        user_input=question,
        response=response,
    )

    print("\nRésultats Ragas :")
    print("Faithfulness :", faithfulness_result)
    print("Answer Relevancy :", relevancy_result)

except Exception as exc:
    print("\nL'évaluation Ragas n'a pas pu être terminée.")
    print("Erreur :", exc)
    print(
        "\nLa cause peut notamment être une limitation "
        "de quota ou de débit de l'API Mistral."
    )