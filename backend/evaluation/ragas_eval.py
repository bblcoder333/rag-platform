import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.retrieval.hybrid_search import hybrid_search
from rag.generator import generate_answer

GOLDEN_DATASET = [
    {
        "question": "What is metamorphic testing?",
        "ground_truth": "Metamorphic Testing (MT) is a novel approach to alleviating the problem of test oracles by verifying the relations between inputs and outputs across multiple test executions."
    },
    {
        "question": "How many LLMs were used in the study?",
        "ground_truth": "Ten LLMs were used in the study, eight private and two public."
    },
    {
        "question": "What datasets were used in the experiment?",
        "ground_truth": "The Conala and MBPP datasets were used in the experiment."
    },
    {
        "question": "What is a metamorphic relation?",
        "ground_truth": "A metamorphic relation (MR) embodies the essential attributes of the target function concerning multiple inputs and their anticipated outputs."
    },
    {
        "question": "What programming language was used in the study?",
        "ground_truth": "Python was used as the programming language in the study."
    }
]


def run_ragas_evaluation():
    print("\nSetting up RAGAS metrics (dict-based API)...")

    llm = LangchainLLMWrapper(Ollama(model="llama3.2"))
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    faithfulness_scores = []
    relevancy_scores = []
    per_question = []

    for item in GOLDEN_DATASET:
        print(f"\nScoring: {item['question']}")

        result = generate_answer(item["question"], top_k=3)
        chunks = hybrid_search(item["question"], top_k=3)
        context_list = [c["text"] for c in chunks]

        row = {
            "question": item["question"],
            "answer": result["answer"],
            "contexts": context_list,
            "ground_truth": item["ground_truth"]
        }

        f_score = asyncio.run(faithfulness.ascore(row))
        r_score = asyncio.run(relevancy.ascore(row))

        print(f"  Faithfulness: {f_score:.3f}  Relevancy: {r_score:.3f}")

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        per_question.append({
            "question": item["question"],
            "faithfulness": f_score,
            "relevancy": r_score
        })

    avg_f = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_r = sum(relevancy_scores) / len(relevancy_scores)

    print("\n" + "="*60)
    print("RAGAS RESULTS")
    print("="*60)
    print(f"  Avg Faithfulness:     {avg_f:.3f}")
    print(f"  Avg Answer Relevancy: {avg_r:.3f}")
    print("="*60)
    print("\nFaithfulness: are answers grounded in retrieved context?")
    print("Relevancy:    does the answer actually address the question?")

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "experiments", "ragas_results.json"
    )
    with open(output_path, "w") as f:
        json.dump({
            "avg_faithfulness": avg_f,
            "avg_answer_relevancy": avg_r,
            "per_question": per_question
        }, f, indent=2)
    print(f"\nResults saved to experiments/ragas_results.json")


if __name__ == "__main__":
    run_ragas_evaluation()