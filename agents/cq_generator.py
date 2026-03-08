import os
import json
import fitz  # PyMuPDF
from typing import List
from pathlib import Path

# LangChain and Pydantic imports
from langchain_core.prompts import PromptTemplate
from langchain_core.schema import HumanMessage
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---- 1. Define Pydantic schema for structured output ----
from models.competency_questions import CompetencyQuestion, CQList

from helper.connections import get_vllm_llm


# ---- 3. Prompt Template ----
def get_prompt_template():
    """Creates a prompt template to generate CQs that drive Ontology Design Patterns."""
    return PromptTemplate(
        template=(
            "You are an expert Ontology Engineer specializing in legal and insurance domains. "
            "Your task is to analyze the document text and generate a list of atomic Competency Questions (CQs) "
            "that will drive the construction of a well-designed ontology using established Ontology Design Patterns (ODPs). "
            "You must return ONLY a JSON object that strictly follows the provided schema.\n\n"
            "**JSON Schema (use EXACT keys and structure):**\n{schema}\n\n"
            "================================================\n"
            "## Guide to Generating ODP-Driven CQs ##\n"
            "================================================\n"
            "Your primary goal is to create CQs that force a rich, reified model structure, not just a flat list of facts. "
            "For each significant piece of information, formulate a set of CQs that break it down according to the patterns below.\n\n"
            "### 1. Event Reification & Participation (Core Pattern)\n"
            "Model actions and events as central classes. Instead of connecting an agent directly to an object, "
            "connect them both to an 'Event' class that captures the context (time, location, manner).\n\n"
            "• **INSTEAD OF THIS (flat model):**\n"
            "  - CQ: “Who can cancel the policy?”\n"
            "• **DO THIS (reified model):** Break the event down into its components:\n"
            "  - CQ 1: “What is the name of the event when a policy's coverage is terminated at the owner's request?” -> odp_hint: 'Event Reification', expected_answer: 'Cancellation'\n"
            "  - CQ 2: “Who is the agent participant in a 'Cancellation' event?” -> odp_hint: 'Participation', expected_answer: 'the Policy Owner'\n"
            "  - CQ 3: “What is the object participant in a 'Cancellation' event?” -> odp_hint: 'Participation', expected_answer: 'the Policy'\n"
            "  - CQ 4: “What is the required manner for initiating a 'Cancellation' event?” -> odp_hint: 'Participation', expected_answer: 'written notice'\n\n"
            "### 2. Quantities and Qualities\n"
            "Treat values like money, time durations, and percentages as classes with their own properties (value and unit).\n\n"
            "• **INSTEAD OF THIS (simple value):**\n"
            "  - CQ: “What is the death benefit?”\n"
            "• **DO THIS (quantity model):** Separate the value from its unit.\n"
            "  - CQ 1: “What is the numeric value of the death benefit?” -> odp_hint: 'Quantity', expected_answer: '500000'\n"
            "  - CQ 2: “What is the currency unit of the death benefit?” -> odp_hint: 'Quantity', expected_answer: 'USD'\n\n"
            "### 3. Time Instants and Intervals\n"
            "Distinguish between points in time (instants) and durations (intervals).\n\n"
            "• **INSTEAD OF THIS (ambiguous time):**\n"
            "  - CQ: “When is the reinstatement period?”\n"
            "• **DO THIS (precise time model):**\n"
            "  - CQ 1: “What is the duration of the reinstatement period?” -> odp_hint: 'Time Interval', expected_answer: '5 years'\n"
            "  - CQ 2: “What event marks the beginning of the reinstatement period interval?” -> odp_hint: 'Time Interval', expected_answer: 'the date of lapse'\n\n"
            "### 4. Situations & States\n"
            "Model the different states a policy can be in (e.g., Active, Lapsed, InGracePeriod) as distinct classes. "
            "Generate CQs about the states themselves and the events that cause transitions between them.\n\n"
            "• **DO THIS (situation model):**\n"
            "  - CQ 1: “What event causes a transition from the 'Active' state to the 'In Grace Period' state?” -> odp_hint: 'Situation', expected_answer: 'a missed premium payment'\n"
            "  - CQ 2: “What is the maximum duration a policy can be in the 'In Grace Period' state?” -> odp_hint: 'Situation', expected_answer: '31 days'\n"
            "  - CQ 3: “What state does the policy enter if a premium is not paid by the end of the Grace Period?” -> odp_hint: 'Situation', expected_answer: 'Lapsed'\n\n"
            "================================================\n"
            "## Final Instructions ##\n"
            "================================================\n"
            "1.  **Deconstruct:** For every important clause, deconstruct it into fine-grained CQs following the ODP guides above.\n"
            "2.  **Be Specific:** Your `expected_answer` must be the precise value from the text.\n"
            "3.  **Complete the Schema:** Fill in all fields for every question: `competency_question`, `key_entities`, `odp_hint`, and `expected_answer`.\n\n"
            "**Document Text:**\n---\n{page_text}\n---\n"
        ),
        input_variables=["page_text"],
        # We pass the new, more detailed schema to the prompt
        partial_variables={"schema": CQList.model_json_schema()}
    )


# ---- 4. PDF Extraction ----
def extract_pages_from_pdf(pdf_path: str):
    """Extracts text from each page of a PDF file."""
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=0):  # Start pages from 0 for consistency
            text = page.get_text("text")
            if text.strip():
                yield i, text


# ---- 5. Main CQ Generation Logic ----
def generate_cqs_for_contract(contract_path: str, output_json: str, structured_llm):
    """
    Processes a single PDF contract, generates CQs for each page,
    and saves the result to a JSON file.
    """
    contract_id = Path(contract_path).stem
    all_results = {contract_id: {}}
    prompt = get_prompt_template()

    print(f"\n📄 Processing contract: {contract_id}")

    for page_number, page_text in extract_pages_from_pdf(contract_path):
        print(f"  - Analyzing page {page_number}...")
        message = HumanMessage(
            content=prompt.format(
                page_text=page_text[:8000]
            )
        )
        try:
            cq_list_object = structured_llm.invoke([message])
            result_dict = cq_list_object.model_dump()
            all_results[contract_id][str(page_number)] = result_dict['questions']
            print(f"  ✔ Successfully parsed page {page_number} with {len(result_dict.get('questions', []))} CQs.")
        except ValidationError as ve:
            print(f"  ❌ [ERROR] Validation failed for page {page_number}: {ve}")
        except Exception as e:
            print(f"  ❌ [ERROR] An unexpected error occurred on page {page_number}: {e}")
            continue

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n💾 Saved all CQs for {contract_id} to {output_json}")


# ---- 6. Main Execution Block ----
if __name__ == "__main__":
    # --- Dynamic Path Configuration ---
    # Assumes the script is run from the project root 'llm-ontology-generation'
    PROJECT_ROOT = Path(__file__).parent.parent
    CONTRACTS_DIR = PROJECT_ROOT / "contracts"
    OUTPUT_DIR = PROJECT_ROOT / "cqs" / "generated"

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CONTRACTS_DIR.exists():
        print(f"🚨 [FATAL ERROR] The contracts directory was not found at: {CONTRACTS_DIR}")
        print("Please ensure the 'contracts' folder exists in your project root.")
    else:
        # --- Initialization ---
        llm = get_vllm_llm()
        structured_chat_llm = llm.with_structured_output(
            schema=CQList,
            method="json_mode"
        )

        # --- Run the Generation Process for all PDFs in the directory ---
        pdf_files = list(CONTRACTS_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"🟡 [WARNING] No PDF files found in {CONTRACTS_DIR}.")

        for contract_file in pdf_files:
            # Define a unique output file for each contract
            output_filename = contract_file.with_suffix(".json").name
            output_path = OUTPUT_DIR / output_filename

            generate_cqs_for_contract(
                contract_path=str(contract_file),
                output_json=str(output_path),
                structured_llm=structured_chat_llm
            )

        print("\n🎉 All contracts processed.")