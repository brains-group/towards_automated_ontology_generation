import argparse
import json
from rag_eval.evaluate_mine import *
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_eval.configurations import *

warnings.filterwarnings("ignore")

# Add the src directory to Python path to import from source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

load_dotenv()

# Assuming OntologyRetriever and evaluate_qa are imported or defined above this
from rag_eval.OntologyRetreiver import OntologyRetriever

# --- 1. Data Transformation ---
def load_and_flatten_cqs(filepath: str) -> list[dict]:
    """
    Loads the nested CQ JSON and flattens it into a list of QA dictionaries.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flat_list = []
    # Iterate through the contracts and subsets (e.g., "0", "1")
    for contract_name, subsets in data.items():
        for subset_id, questions in subsets.items():
            for q in questions:
                flat_list.append({
                    "question": q["competency_question"],
                    "answer": q["expected_answer"]
                })

    print(f"Loaded and flattened {len(flat_list)} competency questions.")
    return flat_list


# --- 2. Parallel Processing Helpers ---
def batch_list(items, max_batch_size=20):
    """Batch a list into chunks of max_batch_size."""
    return [items[i:i + max_batch_size] for i in range(0, len(items), max_batch_size)]


def process_cq_batch(batch_queries, retriever):
    """
    Processes a batch of query dictionaries using the OntologyRetriever.
    """
    batch_results = []
    batch_correct = 0

    for query in batch_queries:
        try:
            # retriever.retrieve returns: node_ids, context_set, context_text
            _, _, context_text = retriever.retrieve(query["question"])

            # Evaluate using your DSPy function
            evaluation = evaluate_qa(
                question=query["question"],
                context=context_text,
                correct_answer=query["answer"]
            )

            result = {
                "query_data": query,
                "retrieved_context": context_text,
                "evaluation": evaluation,
            }
            batch_results.append(result)
            batch_correct += evaluation
        except Exception as e:
            print(f"Error evaluating query '{query.get('question', 'Unknown')}': {e}")

    return batch_results, batch_correct


# --- 3. Core Evaluation Logic ---
def evaluate_cqs_accuracy(retriever, queries: list[dict]):
    """
    Distributes the flattened queries into thread pools for parallel evaluation.
    """
    if not queries:
        return {"accuracy": 0.0, "results": []}

    batches = batch_list(queries, max_batch_size=3)
    print(f"Split {len(queries)} queries into {len(batches)} batches.")

    total_correct = 0
    all_results = []

    # Limit threads to avoid overwhelming the LLM evaluation endpoint
    max_threads = min(len(batches), 24)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_batch = {
            executor.submit(process_cq_batch, batch, retriever): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                batch_results, batch_correct = future.result()
                all_results.extend(batch_results)
                total_correct += batch_correct
                print(f"Batch {batch_index + 1}/{len(batches)} completed.")
            except Exception as exc:
                print(f"Batch {batch_index + 1} generated an exception: {exc}")

    accuracy = total_correct / len(queries)

    print(f"CQ Evaluation complete. Overall accuracy: {accuracy:.2%}")
    return {
        "accuracy": accuracy,
        "results": all_results
    }


def evaluate_cqs(ontology_file: str, cq_file: str, embedding_model: str):
    """
    High-level wrapper to initialize the retriever, load data, and run the eval.
    """
    # 1. Flatten the input JSON
    pairs = load_and_flatten_cqs(cq_file)

    # 2. Initialize our new Ontology Retriever
    print(f"Initializing OntologyRetriever with file: {ontology_file}")
    retriever = OntologyRetriever(
        ontology_file=ontology_file,
        model_name=embedding_model
    )

    # 3. Run parallel evaluation
    results = evaluate_cqs_accuracy(retriever, pairs)
    return results


# --- 4. Saving the Experiment ---
def save_cq_experiment(
        llm_model: str,
        embedding_model: str,
        cq_file: str,
        ontology_file: str,
        contract_name: str,
        output_file: str,
        results: dict,
):
    """Saves the experiment results alongside specific run metadata."""
    metadata = {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "cq_file": cq_file,
        "ontology_file": ontology_file,
        "contract_name": contract_name
    }

    results.update(metadata)

    # Ensure parent directories exist
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Successfully saved CQ experiment for '{contract_name}' to {output_file}")
    except Exception as e:
        print(f"Failed to save experiment to {output_file}. Error: {e}")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    configure_dspy()

    parser = argparse.ArgumentParser(description="Evaluate an ontology against Competency Questions (CQs).")

    parser.add_argument("--ontology-file", type=str, required=True, help="Path to the RDF/OWL ontology file.")
    parser.add_argument("--cq-file", type=str, required=True, help="Path to the competency questions JSON file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the resulting evaluation JSON.")
    parser.add_argument("--contract-name", type=str, default="Equivita Synthetic Life Insurance",
                        help="Name of the contract being evaluated.")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="LLM used for DSPy evaluation.")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for node embeddings.")

    args = parser.parse_args()

    # Note: configure_dspy() should be called here if required by your evaluate_qa setup
    # configure_dspy()

    print(f"Starting CQ evaluation pipeline for contract: {args.contract_name}")

    # Run the evaluation
    eval_results = evaluate_cqs(
        ontology_file=args.ontology_file,
        cq_file=args.cq_file,
        embedding_model=args.embedding_model
    )

    # Save the output
    save_cq_experiment(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        cq_file=args.cq_file,
        ontology_file=args.ontology_file,
        contract_name=args.contract_name,
        output_file=args.output_file,
        results=eval_results
    )

