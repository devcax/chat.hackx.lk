from fastapi import APIRouter
from ..schemas.chat import ChatRequest, ChatResponse
from ..core.rag_handler import query_rag, get_query_engine

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Receives a prompt and returns a RAG-generated response.
    """
    query_engine = get_query_engine()
    response_text = query_rag(chat_request.prompt, query_engine)
    return ChatResponse(response=response_text) 