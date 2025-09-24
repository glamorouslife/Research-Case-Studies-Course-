#!/usr/bin/env python3
import multiprocessing as mp
from typing import Dict, Any
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

@dataclass
class Config:
    model_path: str = r"C:\Users\Toxic\Downloads\BioMistral-7B.Q4_K_S.gguf"
    db_url: str = "http://localhost:6333"
    collection: str = "healthcare_vault_2024"
    embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    max_tokens: int = 2048
    temperature: float = 0.2
    threads: int = mp.cpu_count() // 2
    gpu_layers: int = 8

class HealthcareRAG:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.db = None
        self.qa_chain = None
    
    def _init_llm(self):
        if not self.llm:
            params = {
                'max_new_tokens': self.config.max_tokens,
                'context_length': self.config.max_tokens,
                'repetition_penalty': 1.1,
                'temperature': self.config.temperature,
                'top_k': 50, 'top_p': 1.0, 'stream': True,
                'threads': self.config.threads,
                'gpu_layers': self.config.gpu_layers
            }
            self.llm = CTransformers(
                model=self.config.model_path,
                model_type="mistral", lib="cuda", **params
            )
            print("LLM initialized")
        return self.llm
    
    def _init_db(self):
        if not self.db:
            embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            client = QdrantClient(url=self.config.db_url, prefer_grpc=False)
            self.db = Qdrant(client=client, embeddings=embeddings, collection_name=self.config.collection)
            print("Database connected")
        return self.db
    
    def _init_qa(self):
        if not self.qa_chain:
            prompt_template = """You are a specialized medical advisor. Provide responses based strictly on the given context. If context is insufficient, clearly state limitations.

Medical Context: {context}
Patient Query: {question}
Professional Response:"""
            
            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
            retriever = self.db.as_retriever(search_kwargs={"k": 1})
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever,
                return_source_documents=True, chain_type_kwargs={"prompt": prompt}, verbose=False
            )
        return self.qa_chain
    
    def initialize(self):
        print("Initializing Healthcare RAG system...")
        self._init_llm()
        self._init_db()
        self._init_qa()
        print("System ready")
        return self.qa_chain

def format_response(qa_result):
    docs = qa_result.get('source_documents', [])
    doc = docs[0] if docs else None
    
    response = {
        "answer": qa_result.get('result', ''),
        "source_document": doc.page_content if doc else "",
        "doc": ""
    }
    
    if doc:
        meta = doc.metadata
        response["doc"] = f"{meta.get('source', '')} {meta.get('page', '')} {meta.get('_collection_name', '')}".strip()
    
    return response

# Setup
config = Config()
rag_system = HealthcareRAG(config)

app = FastAPI(title="Healthcare RAG API", description="Medical Knowledge System", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

# Initialize system
qa_engine = rag_system.initialize()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def query(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        print(f"Processing: {query}")
        result = qa_engine(query)
        response = format_response(result)
        print("Query processed")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "processing_failed"}

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "llm": "active",
        "database": "connected",
        "collection": config.collection,
        "embedding": config.embedding_model
    }

@app.get("/info")
async def info():
    return {
        "model": "BioMistral-7B",
        "embeddings": "Microsoft BiomedNLP",
        "database": "Qdrant",
        "gpu_layers": config.gpu_layers,
        "max_tokens": config.max_tokens
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Healthcare RAG server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")