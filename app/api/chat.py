import re
from fastapi import APIRouter
from ..schemas.chat import ChatRequest, ChatResponse
from ..core.rag_handler import query_rag, get_query_engine

router = APIRouter()

def sanitize_response(text: str) -> str:
    """
    Sanitizes the response text by removing unwanted asterisks and forward slashes.
    
    Args:
        text (str): The raw response text from the RAG system
        
    Returns:
        str: Cleaned response text
    """
    # Remove standalone asterisks that appear to be formatting artifacts
    # Keep asterisks that are part of proper markdown formatting (surrounded by text)
    text = re.sub(r'\*{2,}', '', text)  # Remove multiple consecutive asterisks
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)  # Remove asterisks at start of lines
    text = re.sub(r'\s+\*+\s*$', '', text, flags=re.MULTILINE)  # Remove asterisks at end of lines
    text = re.sub(r'\s+\*+\s+', ' ', text)  # Replace asterisks surrounded by spaces with single space
    
    # Remove standalone forward slashes (simple approach)
    # First protect URLs by replacing them temporarily
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    protected_urls = {}
    for i, url in enumerate(urls):
        placeholder = f"__URL_PLACEHOLDER_{i}__"
        protected_urls[placeholder] = url
        text = text.replace(url, placeholder)
    
    # Now remove standalone forward slashes
    text = re.sub(r'\s+/\s+', ' ', text)  # Remove forward slashes surrounded by spaces
    text = re.sub(r'^/\s*', '', text, flags=re.MULTILINE)  # Remove forward slashes at start of lines
    text = re.sub(r'\s*/\s*$', '', text, flags=re.MULTILINE)  # Remove forward slashes at end of lines
    
    # Restore protected URLs
    for placeholder, url in protected_urls.items():
        text = text.replace(placeholder, url)
    
    # Clean up multiple spaces and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Receives a prompt and returns a RAG-generated response.
    """
    query_engine = get_query_engine()
    print(f"Received prompt: {chat_request.prompt}")
    print(f"Using query engine: {query_engine}")
    response_text = query_rag(chat_request.prompt, query_engine)
    
    # Sanitize the response before returning
    sanitized_response = sanitize_response(response_text)
    print(f"Original response: {response_text}")
    print(f"Sanitized response: {sanitized_response}")
    
    return ChatResponse(response=sanitized_response) 