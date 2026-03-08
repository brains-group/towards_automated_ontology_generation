import os
from pathlib import Path
from typing import TypedDict, Annotated

# RDF and OWL libraries for verification
import rdflib
from owlready2 import *

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. Constants and Setup ---
PROJECT_ROOT = Path(__file__).parent.parent
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
ONTOLOGY_DIR.mkdir(exist_ok=True)
MAX_FIX_CYCLES = 1000 # Safety break to prevent infinite loops

# --- 2. Tool Definitions (Unchanged from your version) ---
from tools.file_management import grep_file, write_file_with_range, read_lines_from_file

# These are no longer tools for an agent, but regular functions for our check nodes.
from tools.syntax_checks import verify_rdf_syntax, verify_owl_consistency

import json

class _Clear:
    pass
CLEAR = _Clear()

def message_reducer(existing_messages: list, new_messages: list | _Clear | None) -> list:
    """
    State reducer for messages. Appends new messages, or clears the list
    if it receives the CLEAR sentinel.
    """
    if new_messages is CLEAR:
        # If we get the signal, return a new empty list, clearing the history.
        return []
    if new_messages is None:
        # If a node doesn't return messages, keep the existing ones.
        return existing_messages
    # Otherwise, perform the default append operation.
    return existing_messages + new_messages

def _parse_tool_status(result: str):
    # Handle JSON or plain strings
    try:
        obj = json.loads(result)
        status = obj.get("status", "").lower()
        return status or "error", obj
    except Exception:
        # Fallback for old tools that returned plain strings
        text = result.lower()
        if "success" in text or "✅" in text:
            return "success", {"message": result}
        return "error", {"message": result}


# --- 3. State and Agent Definitions ---
class FixerState(TypedDict):
    """The state for our verification and fixing graph."""
    file_path: str
    error_message: str | None
    result: str
    fix_cycles: int
    messages: Annotated[list, message_reducer]

# --- 4. Node and Graph Logic ---

def syntax_check_node(state: FixerState) -> dict:
    print("\n" + "="*50 + "\nSTEP 1: Checking RDF Syntax\n" + "="*50)
    file_path = state['file_path']
    raw = verify_rdf_syntax.invoke({"file_path": str(file_path)})
    status, payload = _parse_tool_status(raw)
    if status == "success":
        print("✅ RDF Syntax is valid.")
        return {"error_message": None}
    else:
        print(f"❌ RDF Syntax ERROR: {payload.get('message', raw)}")
        return {"error_message": raw, "fix_cycles": state.get('fix_cycles', 0) + 0.01}  # keep raw so the bug fixer sees the JSON

def semantics_check_node(state: FixerState) -> dict:
    print("\n" + "="*50 + "\nSTEP 2: Checking OWL Consistency\n" + "="*50)
    file_path = state['file_path']
    raw = verify_owl_consistency.invoke({"file_path": str(file_path)})
    status, payload = _parse_tool_status(raw)
    if status == "success":
        print("✅ OWL Logic is consistent.")
        return {"error_message": None}
    else:
        print(f"❌ OWL Consistency ERROR: {payload.get('message', raw)}")
        return {"error_message": raw, "fix_cycles": state.get('fix_cycles', 0) + 1}
    
def bug_fixer_node(state: FixerState) -> dict:
    """The agent node that INVOKES THE LLM to fix a reported error."""
    print("\n" + "="*50 + "\nSTEP: Engaging Bug Fixer Agent\n" + "="*50)
    
    system_prompt = """You are an expert programmer responsible for fixing a single, specific error in an ontology file.
You have access to two tools: `read_lines_from_file` and `write_file_with_range`.
Your process is:
1.  Carefully analyze the provided error message.
2. THINK step-by-step about how to fix the error.
3.  Call `read_lines_from_file` to inspect the code and read around the reported lines to get context. You can also use `grep_file` to match specific patterns. DO NOT try to read the entire file. JUST read some lines before and after the error. You might want to read 30 before and 30 after.
4.  Formulate the corrected code block.
5.  Call `write_file_with_range` to replace only the incorrect lines with your fix. Specify the `start_line` and `end_line` precisely.
6. For some errors you will need to make LARGER or SUBSTANTIAL changes and in other errors you will only need to change a few lines.
7.  If your first fix doesn't work, analyze the new error message, re-apply the checklist, read the file again, and try a different fix.

YOU MUST DIAGNOSE THE ISSUE, PROPOSE A FIX, AND THEN APPLY IT.

Once you have read enough lines around the error, you MUST use the `write_file_with_range` tool to write your fix.

--- TURTLE SYNTAX CHECKLIST ---

MISTAKES TO AVOID

Using a prefixed name with an undefined prefix (e.g., ex:Thing without declaring @prefix ex:).

Forgetting . at the end of a triple (subject–predicate–object must end with a period).

Confusing ; and , (; separates predicates for the same subject; , separates multiple objects for the same predicate).

Omitting < > around full IRIs (write <http://example.org/Thing>, not http://example.org/Thing).

Using relative IRIs without a base/@base declaration.

Declaring the same prefix twice with different IRIs unintentionally.

Using : (empty prefix) without explicitly declaring it (many parsers treat this as an error).

Mismatching quotes in string literals ("abc or \"""abc").

Forgetting to escape quotes or backslashes inside strings ("He said \"hi\"").

Mixing language tag and datatype on the same literal ("color"@en^^xsd:string is illegal).

Putting a datatype on a non-string literal incorrectly (42^^xsd:integer is fine; 42.^^xsd:decimal is not).

Writing booleans as bare words in contexts expecting strings, or quoting booleans when a boolean is expected (true/false are boolean literals; "true" is a string).

Using a (shorthand for rdf:type) in subject or object position; a can only be a predicate.

Using a as a prefix (a:Thing) instead of the keyword a.

Leaving a blank node property list [] with no predicates or not attached to a triple.

Misclosing blank nodes or collections (unbalanced [] or ()).

Misusing RDF collections: writing commas inside () or forgetting that () represents an rdf:nil-terminated list.

Trailing ; or , at the end of a subject’s predicate/object list (must end with . not ; or ,).

Using comments incorrectly (only # starts a comment; anything after # to end of line is ignored).

Placing @prefix or @base lines after using those prefixes/relative IRIs (declare before use to avoid parser errors).

Using invalid characters in prefixed local names (spaces, leading -, or illegal punctuation).

Case errors in keywords (@prefix, @base, true, false, a are case-sensitive).

Using ^^ with something that is not an IRI or prefixed name ("x"^^string is invalid; must be "x"^^xsd:string or "x"^^<IRI>).

Forgetting whitespace between tokens, causing accidental concatenation (e.g., ex:s1ex:p).

Ending the file without a final newline or with partial token (some parsers choke on incomplete last line).

Assuming Turtle allows JSON-like objects or commas between triples (it doesn’t; triples end with . only).
"""
    
    prompt = f"""The ontology file at `{state['file_path']}` has a technical error.
--- ERROR MESSAGE ---\n{state['error_message']}\n---------------------
Fix this one error by using your tools. DO NOT make unrelated changes. DO NOT provide explanations or code directly. USE TOOLS ONLY."""

    # --- STRATEGY: Prune the history ---
    # Always include the system prompt and the latest task.
    messages_to_send = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    if (not state['messages']) or len(state['messages']) == 0:
        # If no previous messages, we start fresh
        print("No previous messages. Starting with the initial prompt.")
        state['messages'] = messages_to_send  
    else:
        state['messages'].append(HumanMessage(content="""
        Continue your last task until you are done fixing the error with the tools. If you have read the file and are ready to write, then you MUST USE the `write_file_with_range` tool.
        """))

    # Optional: Add the last 2 messages (one agent turn, one tool response) for context
    # This gives the agent memory of its most recent action.
    
    print("--- 🤖 Bug Fixer is thinking... ---")
    response = llm_with_tools.invoke(state['messages'])  # Use the full state messages for context
    print("--- 💡 Bug Fixer has responded. ---")
    
    # Return the AI's response to be appended to the *full* state for logging
    return {"messages": [response]}

def route_after_check(state: FixerState) -> str:
    """Decision function to route the graph flow."""
    print("\n--- Routing Decision ---")
    if state.get('fix_cycles', 0) >= MAX_FIX_CYCLES:
        print(f"🚫 Max fix cycles ({MAX_FIX_CYCLES}) reached. Aborting.")
        return "end_failure"
        
    if state.get("error_message"):
        print("🚦 Error detected. Routing to Bug Fixer.")
        return "fix_bugs"
    
    # This will be used to decide between semantics_check and end
    print("✅ No error detected. Proceeding to next step.")
    return "continue"

def finalize_success(state: FixerState) -> dict:
    """Final node for a successful run."""
    print("\n" + "="*50 + "\nVERIFICATION COMPLETE\n" + "="*50)
    result_message = f"All checks passed. Ontology '{state['file_path']}' has been verified."
    return {"result": result_message}

def finalize_failure(state: FixerState) -> dict:
    """Final node for a failed run."""
    print("\n" + "="*50 + "\nVERIFICATION FAILED\n" + "="*50)
    result_message = f"Failed to fix ontology '{state['file_path']}' after {state['fix_cycles']} attempts. Last error: {state['error_message']}"
    return {"result": result_message}
    
def clear_messages(state: FixerState) -> dict:
    """Sends a signal to the message_reducer to clear the history."""
    print("Signal sent to clear messages from state.")
    return {"messages": CLEAR}

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    # Create a sample buggy ontology file for demonstration
    
    file_to_fix = "sen_copy.ttl"
    ontology_file_path = ONTOLOGY_DIR / file_to_fix

    # LLM Setup
    print("\nConnecting to local vLLM server...")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    llm = ChatOpenAI(model=model_name, base_url="http://localhost:8000/v1", api_key="not-needed", temperature=0)
    llm_with_tools = llm.bind_tools([read_lines_from_file, write_file_with_range])
    print("✅ Successfully connected to vLLM.")

    # Graph Construction
    builder = StateGraph(FixerState)
    builder.add_node("syntax_check", syntax_check_node)
    builder.add_node("semantics_check", semantics_check_node)
    builder.add_node("bug_fixer", bug_fixer_node)
    # The ToolNode will execute the tool calls returned by the bug_fixer
    builder.add_node("tools", ToolNode([read_lines_from_file, write_file_with_range, grep_file]))
    
    # Nodes for handling the end of the process
    builder.add_node("end_success", finalize_success)
    builder.add_node("end_failure", finalize_failure)
    builder.add_node("clear_messages", clear_messages)

    builder.set_entry_point("syntax_check")

    # Conditional edge after syntax check
    builder.add_conditional_edges(
        "syntax_check",
        route_after_check,
        {"fix_bugs": "bug_fixer", "continue": "semantics_check", "end_failure": "end_failure"}
    )
    
    # Conditional edge after semantics check
    builder.add_conditional_edges(
        "semantics_check",
        route_after_check,
        {"fix_bugs": "bug_fixer", "continue": "end_success", "end_failure": "end_failure"}
    )
    
    # Connect the bug fixer node to the tools node
    builder.add_conditional_edges(
        "bug_fixer",
        tools_condition,
        {"tools": "tools", "__end__": "clear_messages"}
    )
    
    # CRITICAL: After a fix is attempted, ALWAYS go back to the start of the process
    # Now: after tools, check if LLM is done or not
    builder.add_edge("tools", "bug_fixer")  # Loop back to bug fixer
    builder.add_edge("clear_messages", "syntax_check")  # Clear messages and restart the process

    # Connect final nodes to the graph's end
    builder.add_edge("end_success", END)
    builder.add_edge("end_failure", END)
    
    app = builder.compile()

    # Run the graph
    # NEW
    initial_state = {
        "file_path": str(ontology_file_path.resolve()),
        "fix_cycles": 0,
        "messages": []
    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 10000})
    print(f"\nFinal Result: {final_state.get('result')}")
    