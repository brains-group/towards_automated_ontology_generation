import networkx as nx
import numpy as np
import rdflib
import re
import unicodedata
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class OntologyRetriever:
    def __init__(self, ontology_file: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the graph, parses the ontology, and builds the retrieval indexes
        for both nodes AND edges.
        """
        self.graph = nx.DiGraph()
        self.retrieval_model = SentenceTransformer(model_name)

        # Maps an edge relation string to a list of (subject, object) tuples
        self.edge_to_pairs = defaultdict(list)

        # 1. Parse ontology and build networkx graph
        self._build_graph(ontology_file)

        # 2. Store distinct nodes and edges
        self.nodes = list(self.graph.nodes())
        self.edges = list(self.edge_to_pairs.keys())

        # Combine them into a single list for unified indexing and search
        self.search_elements = self.nodes + self.edges
        self.element_types = ["node"] * len(self.nodes) + ["edge"] * len(self.edges)

        # 3. Create embeddings for everything (Nodes + Edges)
        print(f"Creating embeddings for {len(self.nodes)} nodes and {len(self.edges)} edges...")
        self.embeddings = self.retrieval_model.encode(self.search_elements, show_progress_bar=False)

        # 4. Create BM25 index
        print("Building BM25 index...")
        tokenized_elements = [str(el).lower().split() for el in self.search_elements]
        self.bm25 = BM25Okapi(tokenized_elements)

    def _clean_string(self, text: str) -> str:
        """Removes URI prefixes, splits CamelCase, and thoroughly normalizes the string."""
        if "http://" in text or "https://" in text or "urn:" in text:
            text = text.split("#")[-1] if "#" in text else text.split("/")[-1]

        text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).lower().strip()

    def _build_graph(self, ontology_file: str):
        """Helper to parse RDF triples and populate the graph and edge map."""
        print(f"Parsing ontology from {ontology_file}...")
        rdf_graph = rdflib.Graph()
        rdf_graph.parse(ontology_file)

        for s, p, o in rdf_graph:
            s_clean = self._clean_string(str(s))
            p_clean = self._clean_string(str(p))
            o_clean = self._clean_string(str(o))

            if not s_clean or not o_clean or not p_clean:
                continue

            self.graph.add_node(s_clean)
            self.graph.add_node(o_clean)
            self.graph.add_edge(s_clean, o_clean, relation=p_clean)

            # Save the pair so we can easily look up all instances of this edge later
            self.edge_to_pairs[p_clean].append((s_clean, o_clean))

    def get_relevant_elements(self, query: str, top_k: int = 8, mode: str = "node_and_edge") -> list[dict]:
        """Use rank fusion to retrieve top-k elements. Mode controls what gets searched."""
        if mode not in ["node", "node_and_edge"]:
            raise ValueError("Mode must be either 'node' or 'node_and_edge'")

        query_tokens = query.lower().split()
        query_emb = self.retrieval_model.encode([query], show_progress_bar=False)

        bm25_scores = self.bm25.get_scores(query_tokens)
        emb_scores = cosine_similarity(query_emb, self.embeddings).flatten()
        combined_scores = 0*bm25_scores + 10.0*emb_scores

        # If the user only wants nodes, we mask out the edge scores by setting them to -infinity
        if mode == "node":
            node_count = len(self.nodes)
            mask = np.array([True] * node_count + [False] * len(self.edges))
            combined_scores[~mask] = -np.inf

        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        top_items = []
        for i in top_indices:
            if combined_scores[i] == -np.inf:
                continue  # Skip if we hit the masked-out edges

            top_items.append({
                "element": self.search_elements[i],
                "type": self.element_types[i],
                "score": combined_scores[i]
            })

        return top_items

    def retrieve_node_context(self, node: str, depth: int = 2) -> set:
        """Recursively fetches incoming and outgoing edges for a specific node."""
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return

            for neighbor in self.graph.neighbors(current_node):
                rel = self.graph[current_node][neighbor].get("relation", "related to")
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)

            for predecessor in self.graph.predecessors(current_node):
                rel = self.graph[predecessor][current_node].get("relation", "related to")
                context.add(f"{predecessor} {rel} {current_node}.")
                explore_neighbors(predecessor, current_depth + 1)

        if node in self.graph:
            explore_neighbors(node, 1)

        return context

    def retrieve(self, query: str, k: int = 7, mode: str = "node_and_edge", verbose: bool = False):
        """
        Main retrieval function.
        If it finds a node, it expands from that node.
        If it finds an edge, it expands from BOTH the subject and object of every instance of that edge.
        """
        top_matches = self.get_relevant_elements(query, top_k=k, mode=mode)
        context = set()

        for match in top_matches:
            element = match["element"]
            el_type = match["type"]

            if el_type == "node":
                node_context = self.retrieve_node_context(element)
                if verbose:
                    print(f"Context for node '{element}': {node_context}")
                context.update(node_context)

            elif el_type == "edge":
                edge_context = set()
                # Get all (subject, object) pairs connected by this specific edge
                pairs = self.edge_to_pairs.get(element, [])

                for subject_node, object_node in pairs:
                    # Expand outwards from both the subject and the object
                    edge_context.update(self.retrieve_node_context(subject_node))
                    edge_context.update(self.retrieve_node_context(object_node))

                    # Ensure the explicit edge itself is captured
                    edge_context.add(f"{subject_node} {element} {object_node}.")

                if verbose:
                    print(f"Context for edge '{element}' (expanded {len(pairs)} pairs): {edge_context}")
                context.update(edge_context)

        # FIXED: Convert set to a sorted list for deterministic output, and join with newlines
        context_text = "\n".join((list(context)))

        if verbose:
            print(f"Combined context: '{context_text}'\n---")

        return top_matches, context, context_text