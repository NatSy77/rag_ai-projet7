import pandas as pd

from scripts.rag_system import RAGSystem


QUESTIONS_PATH = "evaluation_questions.csv"
RESULTS_PATH = "evaluation_results.csv"


def main():
    """
    Exécute le système RAG sur le jeu de questions
    d'évaluation et sauvegarde les résultats.
    """

    questions_df = pd.read_csv(QUESTIONS_PATH)

    rag = RAGSystem()

    results = []

    for _, row in questions_df.iterrows():
        question = row["question"]
        category = row["category"]

        print(f"\nQuestion : {question}")

        rag_result = rag.ask(question)

        sources = rag_result["sources"]

        source_titles = " | ".join(
            source["title"]
            for source in sources
        )

        results.append(
            {
                "question": question,
                "category": category,
                "answer": rag_result["answer"],
                "source_titles": source_titles,
                "number_of_sources": len(sources),
            }
        )

    results_df = pd.DataFrame(results)
    
    # Colonnes réservées à l'évaluation humaine.
    # Elles permettent de compléter l'évaluation automatique
    # par une appréciation qualitative des réponses du RAG.
    results_df["manual_evaluation"] = ""
    results_df["comment"] = ""

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        f"\nÉvaluation terminée : "
        f"{len(results_df)} questions traitées."
    )

    print(
        f"Résultats sauvegardés dans {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()