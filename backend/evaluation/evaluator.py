import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from rag.retrieval.retriever import retrieve
from rag.generator import generate_answer

# Golden dataset — questions you KNOW the answers to
# This is what you benchmark your system against
GOLDEN_DATASET = [
    {
        "question": "What is this person's GPA?",
        "expected_answer": "3.75",
        "expected_keywords": ["3.75", "GPA"]
    },
    {
        "question": "Where does this person go to university?",
        "expected_answer": "University of Calgary",
        "expected_keywords": ["Calgary", "University"]
    },
    {
        "question": "What degree is this person studying?",
        "expected_answer": "Bachelor of Engineering",
        "expected_keywords": ["Engineering", "Bachelor"]
    },
    {
        "question": "What award did this person win in 2025?",
        "expected_answer": "NSERC Undergraduate Student Research Award",
        "expected_keywords": ["NSERC", "Research", "Award"]
    },
    {
        "question": "What languages does this person speak?",
        "expected_answer": "English, Urdu, and Punjabi",
        "expected_keywords": ["English", "Urdu", "Punjabi"]
    }
]


def evaluate_retrieval(question: str, expected_keywords: list,
                       top_k: int = 5) -> dict:
    """Check if retrieval finds chunks containing expected keywords."""
    chunks = retrieve(question, top_k=top_k)
    combined_text = " ".join([c["text"] for c in chunks]).lower()

    keywords_found = [
        kw for kw in expected_keywords
        if kw.lower() in combined_text
    ]

    recall = len(keywords_found) / len(expected_keywords)
    top_similarity = chunks[0]["similarity"] if chunks else 0

    return {
        "keywords_found": keywords_found,
        "keywords_missing": [k for k in expected_keywords
                             if k not in keywords_found],
        "recall": round(recall, 3),
        "top_similarity": top_similarity,
        "num_chunks_retrieved": len(chunks)
    }


def evaluate_answer(answer: str, expected_keywords: list) -> dict:
    """Check if the generated answer contains expected keywords."""
    answer_lower = answer.lower()
    keywords_found = [
        kw for kw in expected_keywords
        if kw.lower() in answer_lower
    ]
    accuracy = len(keywords_found) / len(expected_keywords)

    return {
        "keywords_found": keywords_found,
        "keywords_missing": [k for k in expected_keywords
                             if k.lower() not in answer_lower],
        "accuracy": round(accuracy, 3)
    }


def run_evaluation(use_reranker: bool = True) -> dict:
    """Run full evaluation suite and return metrics."""
    print(f"\n{'='*60}")
    print(f"Running evaluation (reranker={'ON' if use_reranker else 'OFF'})")
    print(f"{'='*60}\n")

    results = []
    total_retrieval_recall = 0
    total_answer_accuracy = 0
    total_latency = 0

    for i, item in enumerate(GOLDEN_DATASET):
        print(f"[{i+1}/{len(GOLDEN_DATASET)}] {item['question']}")

        start = time.time()

        # Evaluate retrieval
        retrieval_eval = evaluate_retrieval(
            item["question"],
            item["expected_keywords"]
        )

        # Evaluate generation
        result = generate_answer(
            item["question"],
            use_reranker=use_reranker
        )
        latency = time.time() - start

        answer_eval = evaluate_answer(
            result["answer"],
            item["expected_keywords"]
        )

        total_retrieval_recall += retrieval_eval["recall"]
        total_answer_accuracy += answer_eval["accuracy"]
        total_latency += latency

        result_item = {
            "question": item["question"],
            "expected": item["expected_answer"],
            "got": result["answer"][:100],
            "retrieval_recall": retrieval_eval["recall"],
            "answer_accuracy": answer_eval["accuracy"],
            "latency_seconds": round(latency, 2),
            "keywords_missing": answer_eval["keywords_missing"]
        }
        results.append(result_item)

        print(f"  Retrieval recall: {retrieval_eval['recall']}")
        print(f"  Answer accuracy:  {answer_eval['accuracy']}")
        print(f"  Latency:          {latency:.2f}s\n")

    n = len(GOLDEN_DATASET)
    summary = {
        "reranker": use_reranker,
        "total_questions": n,
        "avg_retrieval_recall": round(total_retrieval_recall / n, 3),
        "avg_answer_accuracy": round(total_answer_accuracy / n, 3),
        "avg_latency_seconds": round(total_latency / n, 2),
        "results": results
    }

    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"  Avg retrieval recall: {summary['avg_retrieval_recall']}")
    print(f"  Avg answer accuracy:  {summary['avg_answer_accuracy']}")
    print(f"  Avg latency:          {summary['avg_latency_seconds']}s")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    # Run with reranker ON
    results_with = run_evaluation(use_reranker=True)

    # Run with reranker OFF
    results_without = run_evaluation(use_reranker=False)

    # Compare
    print("\nRERANKER IMPACT:")
    print(f"  Accuracy WITH reranker:    "
          f"{results_with['avg_answer_accuracy']}")
    print(f"  Accuracy WITHOUT reranker: "
          f"{results_without['avg_answer_accuracy']}")
    print(f"  Latency WITH reranker:     "
          f"{results_with['avg_latency_seconds']}s")
    print(f"  Latency WITHOUT reranker:  "
          f"{results_without['avg_latency_seconds']}s")

    # Save results
    with open("experiments/eval_results.json", "w") as f:
        json.dump({
            "with_reranker": results_with,
            "without_reranker": results_without
        }, f, indent=2)
    print("\nResults saved to experiments/eval_results.json")