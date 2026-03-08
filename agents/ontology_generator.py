import os
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Annotated

# PyMuPDF for reading contract files
import fitz 

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
# Using ChatOpenAI to connect to the vLLM server
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Constants and Setup ---
PROJECT_ROOT = Path(__file__).parent.parent
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
ONTOLOGY_DIR.mkdir(exist_ok=True) 
        
from tools.file_management import read_file_with_line_numbers, append_to_file, write_file_with_range
        
tools = [read_file_with_line_numbers, append_to_file, write_file_with_range]

# --- 3. LangGraph State Definition ---
# State definition remains unchanged.
class OntologyGenerationState(TypedDict):
    messages: Annotated[list, add_messages]

# --- 4. The Agentic Graph ---
# The Agent and Graph logic remain unchanged, as LangChain is model-agnostic.
class OntologyEditorAgent:
    def __init__(self, llm):
        self.llm_with_tools = llm.bind_tools(tools)

    def get_graph(self):
        """Returns the compiled LangGraph agent."""
        graph_builder = StateGraph(OntologyGenerationState)
        
        graph_builder.add_node("agent", self.run_agent)
        graph_builder.add_node("tools", ToolNode(tools))
        
        graph_builder.set_entry_point("agent")
        
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
        )
        
        graph_builder.add_edge("tools", "agent")
        return graph_builder.compile()

    def run_agent(self, state: OntologyGenerationState) -> dict:
        """This node invokes the LLM to decide on the next action or to respond."""
        print("--- 🧠 Agent is thinking... ---")
        response = self.llm_with_tools.invoke(state["messages"])
        print("--- 📝 Agent response received ---")
        #print(response)
        return {"messages": [response]}

# --- 5. Main Execution Logic ---
if __name__ == "__main__":
    # --- Model Setup (Unchanged) ---
    print("Connecting to the local vLLM server...")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    
    llm = ChatOpenAI(
        model=model_name,
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        temperature=0,
        max_tokens=4096,
    )
    print("✅ Successfully connected to vLLM.")

    # --- ⬇️ 1. Specify Your Input Files Directly Here ⬇️ ---
    # These paths should point to the files for the SINGLE contract you want to process.
    SPECIFIC_CONTRACT_PATH = PROJECT_ROOT / "contracts" / "sen.pdf"
    SPECIFIC_CQ_PATH = PROJECT_ROOT / "cqs" / "generated" / "Sentinel Synthetic Life Insurance.json"
    # --- ⬆️ ------------------------------------------- ⬆️ ---


    # --- 2. Data Loading and Validation ---
    print(f"📖 Loading CQs from: {SPECIFIC_CQ_PATH}")
    if not SPECIFIC_CQ_PATH.exists():
        raise FileNotFoundError(f"❌ Error: The specified CQ file was not found at {SPECIFIC_CQ_PATH}")
    
    with open(SPECIFIC_CQ_PATH, 'r') as f:
        all_cqs_data = json.load(f)

    if not SPECIFIC_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"❌ Error: The specified contract PDF was not found at {SPECIFIC_CONTRACT_PATH}")

    # CORRECT: Get the contract ID from the PDF filename to ensure a match.
    contract_id = SPECIFIC_CONTRACT_PATH.stem
    print(f"📄 Processing Contract ID: {contract_id}")

    # CORRECT: Look up the CQs for this specific contract_id in the JSON data.
    pages = all_cqs_data.get(contract_id)
    if not pages:
        raise KeyError(f"❌ Error: Contract ID '{contract_id}' not found as a key in the CQ file. Please ensure the PDF filename (without extension) matches a top-level key in the JSON file.")


    # --- 3. Instantiate Agent and Set Up Ontology File ---
    agent_executor = OntologyEditorAgent(llm).get_graph()

    # The ontology file is named after the specific contract being processed.
    ontology_file_name = f"{contract_id}_ontology.ttl"
    ontology_file_path = ONTOLOGY_DIR / ontology_file_name

    # Ensure the ontology file exists with headers before starting
    if not ontology_file_path.exists():
        print(f"File '{ontology_file_name}' not found. Creating with default headers.")
        initial_content = """@prefix : <http://www.example.com/insurance#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""
        ontology_file_path.write_text(initial_content)


    # --- 4. Main Processing Logic (CORRECTED - NO OUTER LOOP) ---
    print(f"\n{'='*50}\n  Processing Contract: {contract_id}\n{'='*50}")

    pdf_doc = None
    try:
        # Directly open the specified PDF file
        pdf_doc = fitz.open(SPECIFIC_CONTRACT_PATH)

        # Loop through the pages for THIS specific contract
        for page_number_str, page_data in pages.items():
            page_number = int(page_number_str)
            print(f"\n--- Starting Page {page_number} ---")
            
            page_text = ""
            if pdf_doc and page_number < len(pdf_doc):
                page_text = pdf_doc[page_number].get_text("text")

            # We just need to check if page_data is a non-empty list.
            if not (isinstance(page_data, list) and page_data):
                print(f"Warning: Malformed or empty CQ data for page {page_number}. Skipping.")
                continue
            
            # page_data IS the list of questions, so we just assign it directly.
            cqs_on_page = page_data
            cq_list_str = "\n".join([f"- {cq['competency_question']}" for cq in cqs_on_page])

            prompt = f"""You are an expert Ontology Engineer working as an autonomous file-editing agent. Your task is to incrementally build a life insurance ontology in Turtle (TTL) format.

Your ONLY method of interaction is by using the provided file system tools. 
You must use the `read_file_with_line_numbers` tool to read the current state of the ontology, and then use `append_to_file` or `write_file_with_range` to add new triples.

A good workflow is to first `read_file_with_line_numbers` to understand its current state, then use `append_to_file` or `write_file_with_range` to add the necessary classes, properties, and individuals or change previous ontology parts based on the page content and CQs.

Your task is to contribute to creating a piece of well-structured ontology by reading information that appeared in the given story, requirements, and restrictions (if there are any).
The way you approach this is first you analyze each CQ and read the given turtle RDF from the tool. Then you add or change the RDF so it can answer the competency questions. Your output at each stage is an append to the previous ones or file edits, just do not repeat.

## ROLE & GOAL ##
You are an expert Ontology Engineer. Your goal is to incrementally update the given Turtle (TTL) ontology file so it can answer the provided Competency Questions (CQs), based on the story and requirements.

## CORE WORKFLOW ##
1.  **Analyze**: Review the CQs and the provided story/context.
2.  **Read**: Use your tools to read the current state of the ontology file.
3.  **Update**: Add or modify the TTL triples to satisfy the CQs. Your output must be tool calls that perform file edits. Do not output raw TTL code as a final answer.

## MODELING INSTRUCTIONS ##

### Classes (Prefix: `Cl_`)
-   **Hierarchy**: Use `rdfs:subClassOf` to create deep, logical class hierarchies.
-   **Definition**: Define classes for general concepts, not specific individuals (e.g., `Cl_Policy` is good, `Cl_Policy123` is not).
-   **Axioms**: Use OWL axioms where appropriate (`owl:equivalentClass`, `owl:disjointWith`, etc.).

### Properties (Object & Data)
-   **Creation**: Define new properties with `owl:Property`. Use `rdfs:subPropertyOf` to create hierarchies.
-   **Object Properties**: Link `Cl_` classes to other `Cl_` classes.
-   **Data Properties**: Link `Cl_` classes to literal data types (e.g., `xsd:string`, `xsd:integer`, `xsd:dateTime`).
-   **Characteristics**: Apply property characteristics when logical (`owl:FunctionalProperty`, `owl:SymmetricProperty`, `owl:TransitiveProperty`, etc.).

### Reification & Design Patterns
-   **Complex Relations**: For any relationship involving more than two classes, or a class and multiple data points, you **must** use reification.
-   **Pivot Classes**: Create an intermediate "pivot" class (e.g., `Cl_PolicyCancellation`) to represent the event or relationship itself.
-   **Connections**: Connect the participating classes (e.g., `Cl_PolicyOwner`, `Cl_Policy`) to the pivot class using object properties. Do NOT use `rdfs:subClassOf` for these connections.

### Restrictions
-   Use OWL restrictions to define class characteristics based on their properties.
-   **`owl:allValuesFrom`**: For constraints where a property *must only* point to instances of a certain class.
-   **`owl:someValuesFrom`**: For constraints where a property *must* point to at least one instance of a certain class.

## KEY PRINCIPLES ##
-   **Incremental Edits**: Do not repeat existing code. Your task is to append new information or modify existing lines.
-   **No Explanations**: Provide only tool calls in your response. Do not include conversational text or comments.

here are some possible mistakes that you might make:
1- forgetting to add prefixes at the beginning of the code.
2- forgetting to write pivot classes at the beginning before starting to code.
3- your output would be concatenated to the previous output rdf, so don't write repetitive words, classes, or ...
4- in your output put all of the previous RDF classes, relations, and restrictions and add yours. your output will be passed to the next stage so don't remove previous code (it is going to replace the previous rdf)
5- you usually forget to write the name of the reification (pivot) that you want to create at the beginning of the output
6- In reification, the reification node (pivot class) is connected to all related classes by object properties, not by the subclassof. it can be a subclass of something, but for reification, it needs object properties.
common mistakes in extracting classes:
1- mistake: not extracting all classes and missing many of them. classes can be found in the story, or in the competency question number and restrictions.
2- Returning empty answer
3- Providing comments or explanations
4- Extracint classes like 'Date', and 'integer' are wrong since they are data properties.
5- not using RDF reification: not extracting pivot classes for modeling relation between classes (more than one class and one data property, or more than two classes)
6- extracting individuals in the text as a class
7- The pivot class is not a sublcass of its components.
common mistakes in the hierarchy extraction:
1- creating an ontology for non-existing classes: creating a new leaf and expanding it into the root
2- returning empty answer or very short
3- Providing comments or explanations
4- Extracting attributes such as date, time, and string names that are related to data properties
5- Forget to add "" around the strings in the tuples
Common mistakes in the object_properties:
1- returning new variables with anything except object_properties
2- returning empty answer or very short
3- providing comments or explanations
4- when the pivot class is created, all of the related classes should point to it (direction of relation is from the classes (domains) 'to'  pivot class (range))
Common mistakes in the data_properties:
1- returning new variables with anything except data_properties
2- returning empty answer or very short
3- providing comments or explanations

AVOID THE MISTAKES ABOVE.

Your primary goal is code reuse and logical consistency. 
Before defining any new class or property, meticulously examine the existing ontology and the Financial Industry Business Ontology (FIBO) for components to reuse or extend using rdfs:subClassOf and rdfs:subPropertyOf. 
When a new concept is unavoidable, design it to be abstract and reusable. 
To structure the knowledge effectively, apply established Ontology Design Patterns (ODPs): use Agent Role to separate an entity from its function (e.g., a person vs. a beneficiary); use Participation and Event Reification to model complex actions as central events with multiple participants; use Quantity to link values with their units (e.g., 500 USD); use Time Interval to define durations and specific points in time; use Situation to model the changing states of entities (e.g., a 'Lapsed Policy'); and use Collection to manage lists of items like required documents. Always ensure your additions integrate seamlessly, creating a cohesive, hierarchical, and logically sound model.

DO NOT MISS IMPORTANT FACTS AND DETAILS FROM THE PAGE TEXT. Make sure to populate the facts onto the ontology file with RDF triples.

these are the prefixes:
@prefix : <http://www.example.org/test#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#>.

**PATH to File to Edit:** `{ontology_file_path}`

**Context from Current Page ({page_number}):**
---
{page_text[:5000]}
---

**Requirements for this page:**
THINK ABOUT WHETHER THE ONTOLOGY YOU ARE CREATING CAN BE USED TO FORM SPARQL QUERIES TO ANSWER THE COMPETENCY QUESTIONS.
THE COMPETENCY QUESTIONS FOR THIS PAGE ARE:
{cq_list_str}

**Your Task:**
Formulate and execute a plan to add the required triples to the ontology file using the available tools. You may call tools multiple times. When you are finished with all the work for this page, provide a final response with no tool calls, for example: "Page {page_number} processed successfully."

When you are done editing the file and writing code, you will be done with your task.
Think about when to append and when to edit because there might be mistakes in some previous content that you need to fix.
Your ONLY method of interaction is by using the provided file system tools. YOU MUST USE THE `read_file_with_line_numbers` TOOL TO READ THE CURRENT STATE OF THE ONTOLOGY, AND THEN USE `append_to_file` OR `write_file_with_range` TO ADD NEW TRIPLES. YOU MUST EDIT THE FILE IN PLACE, DO NOT CREATE A NEW FILE OR RETURN A NEW STRING. YOU WILL NOT USE ANY OTHER TOOLS OR METHODS TO INTERACT WITH THE ONTOLOGY FILE.
"""
            initial_state = { "messages": [HumanMessage(content=prompt)] }
            
            config = {"recursion_limit": 35}
            for event in agent_executor.stream(initial_state, config=config):
                for key, value in event.items():
                    print(f"--- Node: {key} ---")
                    if 'messages' in value:
                         #print(value['messages'][-1].pretty_repr())
                         pass
            
    finally:
        if pdf_doc:
            pdf_doc.close()

    print("\n\n✅ Ontology generation process complete for the specified contract.")