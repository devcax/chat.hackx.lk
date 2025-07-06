import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate

load_dotenv()

# Build paths from the project root to make them robust
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / 'chat_backend' / 'app' / 'data'
STORAGE_DIR = PROJECT_ROOT / 'chat_backend' / 'storage'

# Constants
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3-70b-8192"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
# Load API key from environment or use the hardcoded one
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Configure global settings
Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)
Settings.llm = Groq(model=LLM_MODEL, api_key=GROQ_API_KEY)
Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
Settings.num_output = 512

# Create a custom prompt template
QA_TEMPLATE_STR = (
    "You are mascot, the official hackX AI assistant. You help users with details about hackX 10.0 and its organizing body IMSSA.\n\n"
    "Instructions:\n"
    "- Respond directly to the user's question.\n"
    "- Do not begin with greetings like 'Hi there' or 'Hello'.\n"
    "- Do not add closing phrases like 'Let me know if you need more help'.\n"
    "- Be conversational but efficient and informative.\n"
    "- Never mention you're an AI assistant unless asked explicitly.\n\n"
    "====================\n"
    "KNOWLEDGE BASE:\n"
    "{context_str}\n"
    "====================\n"
    "QUESTION:\n"
    "{query_str}\n"
    "====================\n"
    "ANSWER (as mascot):"
)

QA_TEMPLATE = PromptTemplate(QA_TEMPLATE_STR)


def get_query_engine():
    """
    Loads or creates a LlamaIndex query engine.
    """
    # if os.path.exists(STORAGE_DIR):
    #     print("Forcefully deleting old storage directory...")
    #     shutil.rmtree(STORAGE_DIR)
    #     print("Old storage directory deleted.")

    if not os.path.exists(STORAGE_DIR):
        print("Creating new index...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("Index created and persisted.")
    else:
        print("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded.")

    return index.as_query_engine(
        similarity_top_k=5,
        similarity_cutoff=0.7,
        text_qa_template=QA_TEMPLATE,
    )

def query_rag(prompt: str, query_engine) -> str:
    retriever = query_engine._retriever
    retrieved_nodes = retriever.retrieve(prompt)

    # Lower threshold for very short prompts (likely to be general but still relevant)
    threshold = 0.55 if len(prompt.split()) < 5 else 0.65

    relevant_nodes = [node for node in retrieved_nodes if node.score and node.score >= threshold]

    if not relevant_nodes:
        return "I'm sorry, but I don't have enough information to answer that question."

    response = query_engine.query(prompt)

    return str(response)

 