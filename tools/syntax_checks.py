import os
import json
from pathlib import Path
from langchain_core.tools import tool
import rdflib
from owlready2 import *
from io import BytesIO
from owlrl import DeductiveClosure, OWLRL_Semantics

@tool
def verify_rdf_syntax(file_path: str) -> str:
    """
    Verifies TTL syntax with rdflib.
    Returns JSON with: status, reported_line, reported_col, suspect_line, hint, context (lines).
    """
    print(f"--- TOOL: Verifying RDF syntax for '{file_path}' ---")
    from pathlib import Path
    import json

    p = Path(file_path)
    if not p.exists():
        return json.dumps({"status":"error","message":f"File not found: {file_path}"})

    txt = p.read_text().splitlines()
    def context_block(center, radius=3):
        start = max(1, center - radius)
        end   = min(len(txt), center + radius)
        return [{"line": i, "text": txt[i-1]} for i in range(start, end+1)]

    try:
        g = rdflib.Graph()
        g.parse(str(p), format="turtle")
        print("--- TOOL OUTPUT (verify_rdf_syntax): ✅ RDF syntax is valid.")
        return json.dumps({"status":"success"})
    except Exception as e:
        # Try to extract line/col (rdflib usually includes "at line X, column Y")
        import re
        m_line = re.search(r"line\s+(\d+)", str(e))
        m_col  = re.search(r"column\s+(\d+)", str(e))
        rep_line = int(m_line.group(1)) if m_line else None
        rep_col  = int(m_col.group(1)) if m_col else None

        # Heuristic: suspect the nearest previous unterminated line
        suspect = None
        if rep_line:
            for i in range(rep_line-1, 0, -1):
                s = txt[i-1].strip()
                if not s or s.startswith("#"):  # skip blanks/comments
                    continue
                if not s.endswith(('.', ';', ',', ']', '}')):
                    suspect = i
                    break

        result = {
            "status": "error",
            "message": f"{e}",
            "reported_line": rep_line,
            "reported_col": rep_col,
            "suspect_line": suspect,
            "hint": "If suspect_line is set, check for a missing '.' at end of the previous statement.",
            "context": context_block(rep_line or 1, radius=4) if rep_line else []
        }
        print(f"--- TOOL OUTPUT (verify_rdf_syntax): ❌ {result}")
        return json.dumps(result)

@tool
def verify_owl_consistency_old(file_path: str) -> str:
    """
    Checks for logical inconsistencies in an OWL ontology using Owlready2's reasoner.
    If inconsistent, it returns a detailed error message explaining what to look for.
    """
    print(f"--- TOOL: Verifying OWL consistency for '{file_path}' ---")
    try:
        # Steps 1-4: Parse with rdflib and load into owlready2 (same as before)
        full_path = Path(file_path)
        if not full_path.exists():
            return json.dumps({"status": "error", "message": f"File '{file_path}' not found."})

        g = rdflib.Graph()
        g.parse(str(full_path), format="turtle")
        rdf_xml_string = g.serialize(format="xml")
        in_memory_file = BytesIO(rdf_xml_string.encode('utf-8'))
        world = World()
        onto = world.get_ontology("http://in-memory-ontology/onto.owl").load(fileobj=in_memory_file)

        # Step 5: Run the reasoner and check for inconsistencies
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

        inconsistent_classes = list(onto.inconsistent_classes())

        if not inconsistent_classes:
            print("--- TOOL OUTPUT (verify_owl_consistency): ✅ Ontology is logically consistent.")
            return json.dumps({"status": "success", "message": "Ontology is logically consistent."})
        else:
            # --- THIS IS THE KEY CHANGE ---
            # Create a detailed, actionable error message for the LLM agent.
            class_names = [c.name for c in inconsistent_classes]
            error_message = (
                f"The ontology has a logical contradiction. The class(es) '{', '.join(class_names)}' are inconsistent. "
                "This is NOT a simple syntax error. It means the class is defined in a contradictory way, "
                "such as being a subclass of two other classes that are declared to be 'owl:disjointWith' each other. "
                "To fix this, you must find all axioms related to the inconsistent class(es) and their parent classes and resolve the conflict."
            )
            print(f"--- TOOL OUTPUT (verify_owl_consistency): ❌ {error_message}")
            # Return a structured JSON object as a string
            return json.dumps({
                "status": "error",
                "message": error_message,
                "context": {"inconsistent_classes": class_names}
            })

    except Exception as e:
        error_message = f"An unexpected error occurred during consistency check. Details: {e}"
        print(f"--- TOOL OUTPUT (verify_owl_consistency): ❌ {error_message}")
        return json.dumps({"status": "error", "message": error_message})
        
@tool
def verify_owl_consistency(file_path: str) -> str:
    """
    Broad OWL TTL validator using Owlready2 + HermiT.
    Detects unsatisfiable classes, inferred inconsistencies, and conflicting axioms.
    Avoids hardcoded owl.Nothing and works generically.
    """
    print(f"--- TOOL: Verifying OWL consistency for '{file_path}' (Pellet/HermiT) ---")
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({
                "status": "error",
                "message": f"File not found: {file_path}"
            })

        # Load TTL via RDFLib
        g = rdflib.Graph()
        g.parse(str(path), format="turtle")

        rdfxml = g.serialize(format="xml")
        rdf_stream = BytesIO(rdfxml.encode("utf-8"))

        # Load RDF/XML into Owlready2
        world = World()
        onto = world.get_ontology("http://temp.org/temp.owl").load(fileobj=rdf_stream)

        # Ensure entities are initialized
        _ = list(onto.classes())
        _ = list(onto.individuals())
        _ = list(onto.object_properties())

        # Run reasoner
        with onto:
            sync_reasoner_hermit()

        issues = {}

        # 1. Inconsistent classes
        inconsistent = list(onto.inconsistent_classes())
        if inconsistent:
            issues["inconsistent_classes"] = [cls.name for cls in inconsistent]

        # 2. Inferred owl:Nothing types from any individual
        nothing_uri = "http://www.w3.org/2002/07/owl#Nothing"
        inferred_nothing = []
        for ind in onto.individuals():
            types = ind.is_a
            if any(str(t.iri) == nothing_uri for t in types):
                inferred_nothing.append(ind.name)
        if inferred_nothing:
            issues["individuals_in_owl_Nothing"] = inferred_nothing

        # 3. Conflicting property characteristics
        conflicting_props = []
        for prop in onto.object_properties():
            types = {cls.name for cls in prop.is_a}
            if "SymmetricProperty" in types and "AsymmetricProperty" in types:
                conflicting_props.append(prop.name)
        if conflicting_props:
            issues["conflicting_property_characteristics"] = conflicting_props

        # Final result
        if issues:
            print(f"--- TOOL OUTPUT: ❌ Detected logical issues")
            return json.dumps({
                "status": "error",
                "message": "Logical inconsistencies detected in ontology.",
                "context": issues
            })
        else:
            print("--- TOOL OUTPUT: ✅ Ontology is logically consistent.")
            return json.dumps({
                "status": "success",
                "message": "Ontology is logically consistent."
            })

    except Exception as e:
        err = f"Unexpected error during consistency check. Details: {e}"
        print(f"--- TOOL OUTPUT: ❌ Unexpected error: {err}")
        return json.dumps({"status": "error", "message": err})