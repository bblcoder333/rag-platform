import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nest_asyncio
nest_asyncio.apply()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.retrieval.hybrid_search import hybrid_search
from rag.generator import generate_answer
import json

# Golden dataset with reference answers
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
    print("\nSetting up RAGAS evaluation...")
    print("This will take a few minutes — RAGAS judges each answer with an LLM\n")

    # Use Ollama as the judge LLM
    ollama_llm = LangchainLLMWrapper(
        Ollama(model="llama3.2", temperature=0)
    )

    # Use local embeddings
    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )
    )

    # Build evaluation dataset
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in GOLDEN_DATASET:
        print(f"Generating answer for: {item['question']}")

        # Get answer from our RAG system
        result = generate_answer(item["question"], top_k=3)

        # Get retrieved contexts
        chunks = hybrid_search(item["question"], top_k=3)
        context_list = [c["text"] for c in chunks]

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(context_list)
        ground_truths.append(item["ground_truth"])

    # Create RAGAS dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("\nRunning RAGAS evaluation...")
    print("Metrics: faithfulness, answer_relevancy, context_recall, context_precision\n")

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ],
        llm=ollama_llm,
        embeddings=hf_embeddings
    )

    print("\n" + "="*60)
    print("RAGAS RESULTS")
    print("="*60)
    print(f"  Faithfulness:        {result['faithfulness']:.3f}")
    print(f"  Answer Relevancy:    {result['answer_relevancy']:.3f}")
    print(f"  Context Recall:      {result['context_recall']:.3f}")
    print(f"  Context Precision:   {result['context_precision']:.3f}")
    print("="*60)

    print("\nWhat these mean:")
    print("  Faithfulness:      Are answers grounded in retrieved context? (no hallucination)")
    print("  Answer Relevancy:  Does the answer actually address the question?")
    print("  Context Recall:    Did retrieval find all necessary information?")
    print("  Context Precision: Are retrieved chunks relevant to the question?")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "experiments", "ragas_results.json"
    )
    with open(output_path, "w") as f:
        json.dump({
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_recall": result["context_recall"],
            "context_precision": result["context_precision"]
        }, f, indent=2)
    print(f"\nResults saved to experiments/ragas_results.json")

    return result


if __name__ == "__main__":
    run_ragas_evaluation()