import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

# --- Load Environment Variables ---
# Make sure you have a .env file with your GROQ_API_KEY
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# --- Configuration ---
# Define constants for models and pipeline settings
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama3-70b-8192"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
RELEVANCE_THRESHOLD = 0.5  # Minimum score for a document to be considered relevant

# Define paths for data and storage using absolute paths based on this file location
# app/core/rag_handler.py -> parent is app/, data is app/data, storage is app/storage
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = str((SCRIPT_DIR.parent / 'data').resolve())
STORAGE_DIR = str((SCRIPT_DIR.parent / 'storage').resolve())

# --- Global Settings for LlamaIndex ---
# Configure the components that LlamaIndex will use
Settings.embed_model = FastEmbedEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm = Groq(model=LLM_MODEL_NAME, api_key=GROQ_API_KEY)
Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
Settings.num_output = 512 # Sets the maximum number of tokens for the response

# --- Custom Prompt Template ---
# This template tells the AI assistant how to behave.
# It's given strict rules to only answer questions about specific topics.
QA_TEMPLATE_STR = (
    "You are Mascot, the official hackX AI assistant. You ONLY answer questions about hackX 10.0\n\n"
    "Hard rules (must follow):\n"
    "- Answer strictly using the information in the KNOWLEDGE BASE context.\n"
    "- If the answer is not present in the context or the question is unrelated to hackX, respond exactly: \n"
    "  'I can only help with hackX 10.0\n If you have any further questions, please refer to the official hackX 10.0 rule book or contact our coordinators.'\n"
    "- Do not use general world knowledge or guess.\n"
    "- Do not start with greetings or add closing phrases.\n"
    "- Be concise and informative.\n\n"
    "====================\n"
    "KNOWLEDGE BASE:\n"
    "{context_str}\n"
    "====================\n"
    "QUESTION:\n"
    "{query_str}\n"
    "====================\n"
    "Final answer (as Mascot):"
)
QA_TEMPLATE = PromptTemplate(QA_TEMPLATE_STR)

# --- Message for out-of-scope questions ---
OUT_OF_SCOPE_MESSAGE = "I can only help with hackX 10.0\n If you have any further questions, please refer to the official hackX 10.0 rule book or contact our coordinators."


def get_query_engine():
    """
    Loads or creates the search index and prepares the query engine.
    """
    # If the storage directory doesn't exist, we need to build the index
    if not os.path.exists(STORAGE_DIR):
        print(f"Storage not found. Creating new index from documents in '{DATA_DIR}'...")
        # Make sure the data directory exists
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
             raise FileNotFoundError(
                 "Data directory not found or empty: "
                 f"{DATA_DIR}. Place your source files there (Markdown .md, text .txt, or PDF .pdf are supported)."
             )
        
        # Load documents from the data directory
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        
        # Create the vector store index from the documents
        index = VectorStoreIndex.from_documents(documents)
        
        # Persist the index to disk for future use
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("Index created and saved.")
    else:
        # If the index already exists, load it from disk
        print(f"Loading existing index from '{STORAGE_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded.")

    # Create a query engine with our custom prompt template
    query_engine = index.as_query_engine(
        text_qa_template=QA_TEMPLATE,
        similarity_top_k=3  # Retrieve the top 3 most similar documents
    )
    
    return query_engine

def is_malicious_input(prompt: str) -> bool:
    """Check for verbatim/repeat commands or recursive patterns."""
    forbidden_patterns = [
        "repeat and print",
        "verbatim",
        "above text",
        "previous message",
        "copy this",
        "echo this"
    ]
    prompt_lower = prompt.lower()
    return any(pattern in prompt_lower for pattern in forbidden_patterns)


def query_rag(prompt: str, query_engine) -> str:
    """
    Queries the RAG pipeline after checking if the retrieved context is relevant.
    """

    if is_malicious_input(prompt):
        return OUT_OF_SCOPE_MESSAGE
    # First, retrieve the most relevant documents (nodes) for the prompt.
    retrieved_nodes = query_engine._retriever.retrieve(prompt)

    # If the retriever found no documents, the topic is likely out of scope.
    if not retrieved_nodes:
        return OUT_OF_SCOPE_MESSAGE
        
    # Check if the score of the most relevant node is above our threshold.
    # The 'score' indicates how similar the document is to the question.
    top_node = retrieved_nodes[0]
    if top_node.score < RELEVANCE_THRESHOLD:
        print(f"Query '{prompt[:30]}...' is likely out of scope. Top score: {top_node.score:.2f}")
        return OUT_OF_SCOPE_MESSAGE

    # If the context is relevant enough, send the query to the LLM.
    print(f"Querying LLM with relevant context (top score: {top_node.score:.2f})...")
    response = query_engine.query(prompt)
    
    return str(response)



