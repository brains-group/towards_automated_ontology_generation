import os
import json
import argparse
from pathlib import Path
from typing import TypedDict, List, Dict, Optional, Annotated

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# RDFLib for syntax validation and query execution
import rdflib
import rdflib.plugins.sparql.parser

from dotenv import load_dotenv
import traceback
import time

from helper.connections import connect_to_vllm

# --- 1. Constants and Setup ---
MAX_CYCLES = 15
MAX_RETRIES = 5  # still used elsewhere if you want; generator now uses fixed 3 tries per your requirement

# --- Helper: non-throwing invoke with fixed 3 tries for generator ---
def try_invoke_structured(model, messages, tries: int = 3):
    """
    Attempt up to `tries` LLM invocations.
    Returns (response, error). Never raises.
    If all tries fail, response is None and error is the last exception.
    """
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            return model.invoke(messages), None
        except Exception as e:
            last_err = e
            print(f"[Try {attempt}/{tries}] LLM invocation failed: {type(e).__name__} - {e}")
            time.sleep(0.6)
    return None, last_err

# --- 2. Pydantic Models for Structured Output ---
class SparqlQuery(BaseModel):
    """A Pydantic model to ensure the LLM outputs a clean SPARQL query."""
    query: str = Field(description="The syntactically correct SPARQL query.")

class EvaluationResult(BaseModel):
    """Structured model for the final evaluation of a competency question."""
    justification: str = Field(description="A brief, clear justification for the assigned score, comparing the expected answer to the actual query result. If the query was wrong, explain why.")
    score: float = Field(description="A score from the rubric: 1.0 (Correct), 0.5 (Partially Correct), 0.25 (Partial Success), or 0.0 (Incorrect/Error).")

# --- 3. State Definition for the Graph ---
class EvaluationState(TypedDict):
    """Represents the state of the self-correcting evaluation workflow."""
    competency_question: Dict
    ontology_content: str
    messages: Annotated[list, lambda x, y: x + y]
    cycles: int
    generated_query: SparqlQuery
    syntax_error: Optional[str]
    execution_error: Optional[str]
    query_result: Optional[str]
    final_evaluation: Optional[EvaluationResult]
    hard_fail: Optional[bool]   # <-- NEW: route to failure if generator fails 3 times

# --- 4. Agent and Node Definitions ---
llm = connect_to_vllm()
structured_llm_query = llm.with_structured_output(SparqlQuery)

# MODIFIED: query_generator_node now hard-fails after 3 exceptions (any exception)
def query_generator_node(state: EvaluationState) -> Dict:
    """Agent that generates or refines a SPARQL query based on state."""
    print("--- 📝 QueryGenerator Agent: Generating SPARQL... ---")
    
    # Add context from previous failures if they exist
    if state.get("syntax_error"):
        error_msg = (
            "Your previous query failed with a syntax error. "
            "Analyze the error and generate a new, valid query. "
            f"Error: {state['syntax_error']}"
        )
        state["messages"].append(HumanMessage(content=error_msg))
    elif state.get("execution_error"):
        error_msg = (
            "Your query was syntactically valid but failed during execution. "
            "This likely means the entities or properties in the query do not exist in the ontology. "
            "Re-examine the schema and the error message to write a corrected query. "
            f"Error: {state['execution_error']}"
        )
        state["messages"].append(HumanMessage(content=error_msg))
    elif state.get("final_evaluation") and state["final_evaluation"].score == 0.0:
        justification = state["final_evaluation"].justification
        error_msg = (
            "Your previous query executed successfully, but the result was judged as incorrect (Score: 0.0). "
            f"The evaluator's justification was: '{justification}'. Please analyze this feedback, re-read the "
            "question and schema carefully, and generate a new query that produces the correct answer."
        )
        state["messages"].append(HumanMessage(content=error_msg))

    # >>> Only up to 3 tries; on any exception, retry. After 3, hard-fail.
    response, err = try_invoke_structured(structured_llm_query, state["messages"], tries=3)
    if err or response is None:
        print("--- ROUTER: Query generation hard-failed after 3 tries. ---")
        eval0 = EvaluationResult(
            justification=f"Query generation failed after 3 attempts: {type(err).__name__}: {err}" if err else
                         "Query generation returned no response after 3 attempts.",
            score=0.0
        )
        return {
            "hard_fail": True,
            "final_evaluation": eval0
        }

    ai_response = AIMessage(content=response.query)
    print("--- ROUTER: Generated SPARQL query.")
    print(f"Generated Query: {response.query}")

    return {
        "generated_query": response,
        "syntax_error": None,
        "execution_error": None,
        "final_evaluation": None,  # Clear previous evaluation
        "messages": [ai_response],
        "hard_fail": False
    }

def syntax_validator_node(state: EvaluationState) -> Dict:
    """A non-LLM node that validates SPARQL syntax using RDFLib."""
    print("--- 🔍 SyntaxValidator Node: Checking syntax... ---")
    query_str = state["generated_query"].query
    try:
        rdflib.plugins.sparql.parser.parseQuery(query_str)
        print("--- ROUTER: Syntax valid.")
        return {"syntax_error": None}
    except Exception as e:
        print("--- ROUTER: Syntax invalid, looping back.")
        error_message = f"Invalid SPARQL syntax. Details: {e}"
        return {"syntax_error": error_message, "cycles": state["cycles"] + 1}

def query_executor_node(state: EvaluationState) -> dict:
    """A non-LLM node that executes the SPARQL query using RDFLib."""
    print("--- ⚡ QueryExecutor Node: Running query... ---")
    query_str = state["generated_query"].query
    ontology_content = state["ontology_content"]
    
    try:
        g = rdflib.Graph()
        g.parse(data=ontology_content, format="turtle")
        results = g.query(query_str)
        
        if results.type == 'SELECT':
            results_list = [row.asdict() for row in results]
            results_json = json.dumps(results_list, indent=2)

        elif results.type == 'ASK':
            results_json = json.dumps({"boolean": results.askAnswer}, indent=2)

        elif results.type in ('CONSTRUCT', 'DESCRIBE'):
            results_json = results.serialize(format='turtle')
            
        else:
            results_json = "Unknown query result type."

        print("--- ROUTER: Query executed successfully.")
        return {"query_result": results_json, "execution_error": None}

    except Exception:
        print("--- ROUTER: Query execution failed, looping back.")
        error_message = f"Query execution failed. See details below:\n\n{traceback.format_exc()}"
        print(error_message)
        return {"execution_error": error_message, "cycles": state["cycles"] + 1}

def answer_evaluator_node(state: EvaluationState) -> Dict:
    """The 'Judge' agent that scores the final result."""
    print("--- ⚖️ AnswerEvaluator Agent: Judging the result... ---")
    
    structured_llm_eval = llm.with_structured_output(EvaluationResult)
    cq = state["competency_question"]

    prompt = f"""You are a strict and fair evaluator. Your task is to assess if the result of a SPARQL query correctly answers a given question.

    **Scoring Rubric:**
    - **1.0 (Correct):** The query result contains the information needed to confirm the expected answer.
    - **0.X (Partially Correct):** The query RESULT is related but incomplete or contains some incorrect data. You decide the exact score from 0.0 to 0.99 based on how close it is to being correct. DO NOT grade based on just query syntax but mainly judge on the actual result.
    - **0.0 (Incorrect):** The query result is completely wrong, or it returned no results when an answer was expected.

    **Evaluation Task:**
    - **Original Question:** "{cq['competency_question']}"
    - **Expected Answer:** "{cq['expected_answer']}"
    - **Generated SPARQL Query:** ```sparql
    {state["generated_query"].query}
    ```
    - **Actual Query Result (JSON):**
    ```json
    {state['query_result']}
    ```

    Based on the rubric, provide your evaluation. In your justification, briefly comment on why the SPARQL query was correct or incorrect for the task.
    """
    
    evaluation = structured_llm_eval.invoke([HumanMessage(content=prompt)])
    
    if evaluation.score == 0.0:
        print("--- ROUTER: Answer is incorrect, looping back.")
        return {"final_evaluation": evaluation, "cycles": state["cycles"] + 1}
    else:
        print("--- ROUTER: Answer is correct or partially correct, ending.")
        return {"final_evaluation": evaluation}

def handle_failure_node(state: EvaluationState) -> Dict:
    """Node to handle failures after exceeding the max retry limit or hard-fail from generator."""
    print(f"--- ❌ FAILURE HANDLER ---")
    # If generator hard-failed, keep its 0.0 evaluation; otherwise compute based on cycles/result
    if state.get("final_evaluation") is not None and state["final_evaluation"].score == 0.0:
        return {"final_evaluation": state["final_evaluation"]}
    justification = f"Agent failed to produce a correct answer after {state['cycles']} attempts."
    score = 0.25 if state.get("query_result") is not None else 0.0
    if score > 0:
        justification += " Partial credit (0.25) awarded because at least one query was successfully executed."
    evaluation = EvaluationResult(justification=justification, score=score)
    return {"final_evaluation": evaluation}

# --- 5. Graph Definition ---
def build_graph():
    """Builds and compiles the self-correcting evaluation LangGraph."""
    workflow = StateGraph(EvaluationState)
    
    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("syntax_validator", syntax_validator_node)
    workflow.add_node("query_executor", query_executor_node)
    workflow.add_node("answer_evaluator", answer_evaluator_node)
    workflow.add_node("handle_failure", handle_failure_node)
    
    workflow.set_entry_point("query_generator")

    # NEW: route out of generator; go straight to failure when hard_fail=True
    def route_after_generation(state: EvaluationState):
        return "failure" if state.get("hard_fail") else "syntax"

    workflow.add_conditional_edges(
        "query_generator",
        route_after_generation,
        {"syntax": "syntax_validator", "failure": "handle_failure"}
    )
    
    def route_after_syntax(state: EvaluationState):
        if state["cycles"] >= MAX_CYCLES: return "failure"
        return "execute" if state["syntax_error"] is None else "regenerate"

    def route_after_execution(state: EvaluationState):
        if state["cycles"] >= MAX_CYCLES: return "failure"
        return "evaluate" if state["execution_error"] is None else "regenerate"

    def route_after_evaluation(state: EvaluationState):
        if state["cycles"] >= MAX_CYCLES: return "failure"
        return "success" if state["final_evaluation"].score > 0.0 else "regenerate"

    workflow.add_conditional_edges(
        "syntax_validator", route_after_syntax,
        {"regenerate": "query_generator", "execute": "query_executor", "failure": "handle_failure"}
    )
    workflow.add_conditional_edges(
        "query_executor", route_after_execution,
        {"regenerate": "query_generator", "evaluate": "answer_evaluator", "failure": "handle_failure"}
    )
    workflow.add_conditional_edges(
        "answer_evaluator", route_after_evaluation,
        {"regenerate": "query_generator", "success": END, "failure": "handle_failure"}
    )
    workflow.add_edge("handle_failure", END)
    
    return workflow.compile()

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ontology against competency questions.")
    parser.add_argument("cq_file", type=str, help="Path to the competency questions JSON file.")
    parser.add_argument("ontology_file", type=str, help="Path to the ontology TTL file.")
    args = parser.parse_args()

    cq_path = Path(args.cq_file)
    ontology_path = Path(args.ontology_file)
    
    if not cq_path.exists() or not ontology_path.exists():
        print("❌ Error: One or both specified file paths do not exist.")
        exit()

    print("🤖 Initializing Ontology Grader...")
    print(f"▶️  Using Ontology: {ontology_path.name}")
    print(f"▶️  Using CQs:      {cq_path.name}")

    # Load Data
    with open(cq_path, 'r', encoding='utf-8') as f:
        all_cqs_data = json.load(f)
    ontology_content = ontology_path.read_text(encoding='utf-8')
    
    app = build_graph()
    all_results = []
    total_score = 0
    
    cqs_to_process = []
    for top_level_key, pages in all_cqs_data.items():
        if isinstance(pages, dict):
            for page_num, page_data in pages.items():
                cqs_to_process.extend(page_data)
        elif isinstance(pages, list):
            cqs_to_process.extend(pages)

    # Main Evaluation Loop
    for i, cq_item in enumerate(cqs_to_process):
        print(f"\n{'='*60}\nEvaluating CQ {i+1}/{len(cqs_to_process)}: \"{cq_item['competency_question']}\"\n{'='*60}")
        
        initial_prompt = f"""You are an expert in SPARQL and OWL ontologies. Your task is to convert a natural language question into a SPARQL query based on the provided ontology schema.

        **Ontology Schema (TTL format):**
        ```turtle
        {ontology_content}
        ```
        **Natural Language Question:**
        "{cq_item['competency_question']}"
        
        **Expected Answer:**
        "{cq_item['expected_answer']}"

        Based ONLY on the schema, write the corresponding SPARQL query. Output ONLY the SPARQL query inside a JSON object.
        """
        initial_state = {
            "competency_question": cq_item,
            "ontology_content": ontology_content,
            "messages": [HumanMessage(content=initial_prompt)],
            "cycles": 0,
        }
        
        print(f"--- Starting evaluation for CQ {i+1} ---")
        
        graph_config = {"recursion_limit": 10000}
        final_state = app.invoke(initial_state, config=graph_config)
        
        result_entry = {
            "competency_question": cq_item["competency_question"],
            "expected_answer": cq_item["expected_answer"],
            "generated_sparql": final_state["generated_query"].query if final_state.get("generated_query") else "N/A",
            "query_result": json.loads(final_state["query_result"]) if final_state.get("query_result") else "N/A",
            "evaluation_summary": final_state["final_evaluation"].justification,
            "score": final_state["final_evaluation"].score,
        }
        
        print("="*50)
        print(f"\n-> Final Result for CQ {i+1}: [Score: {result_entry['score']}] - {result_entry['evaluation_summary']}")
        print("="*50)
        
        all_results.append(result_entry)
        total_score += result_entry["score"]

    # Save and Print Final Report
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "eval_results"
    results_dir.mkdir(exist_ok=True)
    report_file_name = f"evaluation_report_{ontology_path.stem}.json"
    report_file_path = results_dir / report_file_name
    
    report_data = {
        "ontology_file": ontology_path.name,
        "cq_file": cq_path.name,
        "average_score": total_score / len(cqs_to_process) if cqs_to_process else 0,
        "results": all_results
    }
    
    with open(report_file_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
        
    average_score = total_score / len(cqs_to_process) if cqs_to_process else 0
    
    print(f"\n\n{'='*60}")
    print("🏁 EVALUATION COMPLETE 🏁")
    print(f"{'='*60}")
    print(f"Total CQs Evaluated: {len(cqs_to_process)}")
    print(f"Average Score: {average_score:.2f}")
    print(f"Detailed report saved to: {report_file_path}")
    print(f"{'='*60}")
