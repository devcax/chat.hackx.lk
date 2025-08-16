import re
from fastapi import APIRouter
from ..schemas.chat import ChatRequest, ChatResponse
from ..core.rag_handler import query_rag, get_query_engine

router = APIRouter()

def sanitize_response(text: str) -> str:
    import re

    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove ANSI escape codes (if any)
    text = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

    # Collapse runs of spaces/tabs but KEEP newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # Trim trailing spaces at end of lines
    text = re.sub(r'[ \t]+\n', '\n', text)

    # Allow at most two blank lines in a row (keeps paragraphs/lists readable)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


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