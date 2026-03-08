from langchain_openai import ChatOpenAI  # Use ChatOpenAI for vLLM
import os
from dotenv import load_dotenv

# Load the environment
load_dotenv()

def connect_to_vllm(model_name: str = None, temperature: float = 0.3, max_tokens: int = 4096, port: int = 8000) -> ChatOpenAI:
    """Initializes and returns the ChatOpenAI client for the vLLM server."""
    print("🔌 Connecting to the local vLLM server...")

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

    llm = ChatOpenAI(
        model=model_name,
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print("✅ Successfully connected to vLLM.")
    return llm
    
if __name__ == "__main__":
    connect_to_vllm()