from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import chat as chat_api

app = FastAPI(
    title="Chat API",
    description="An API for a RAG-based chatbot.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.include_router(chat_api.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the hackx Chat API"} 