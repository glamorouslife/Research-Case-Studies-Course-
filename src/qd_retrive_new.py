#!/usr/bin/env python3
"""Healthcare document processing and retrieval system"""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, GoogleDriveLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantStore
from qdrant_client import QdrantClient

load_dotenv()

class HealthcareHub:
    def __init__(self, gdrive_creds_path: str = None):
        self.embed_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.db_url = "http://localhost:6333"
        self.vault_name = "healthcare_vault_2024"
        self.embedder = None
        self.client = None
        self.knowledge_base = None
        self.gdrive_creds_path = gdrive_creds_path
    
    def _get_embedder(self):
        if not self.embedder:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model)
        return self.embedder
    
    def _get_client(self):
        if not self.client:
            self.client = QdrantClient(url=self.db_url, prefer_grpc=False)
        return self.client
    
    def _get_store(self):
        if not self.knowledge_base:
            self.knowledge_base = Qdrant(
                client=self._get_client(),
                embeddings=self._get_embedder(),
                collection_name=self.vault_name
            )
        return self.knowledge_base
    
    def load_docs(self, docs_path="Data"):
        if Path(docs_path).exists():
            print(f"Loading documents from local path: {docs_path}")
            loader = DirectoryLoader(docs_path, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        else:
            print(f"Loading documents from Google Drive folder ID: {docs_path}")
            loader = GoogleDriveLoader(
                folder_id=docs_path,
                recursive=False,
                file_loader_cls=PyPDFLoader
            )
        
        raw_docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120,
            separators=["\n\n", "\n", ".", " "]
        )
        return splitter.split_documents(raw_docs)
    
    def build_vault(self, chunks):
        vault = QdrantStore.from_documents(
            chunks, self._get_embedder(),
            url=self.db_url, collection_name=self.vault_name, force_recreate=True
        )
        return vault
    
    def search(self, question, top_k=2):
        return self._get_store().similarity_search_with_score(query=question, k=top_k)
    
    def display(self, results, preview_len=250):
        print(f"\nFound {len(results)} relevant sources:")
        print("=" * 60)
        
        for idx, (doc, score) in enumerate(results, 1):
            content = doc.page_content[:preview_len]
            if len(doc.page_content) > preview_len:
                content += "..."
            
            print(f"\nSource #{idx} | Relevance: {score:.3f}")
            print(f"Content: {content}")
            print(f"Meta: {doc.metadata}")
            print("-" * 40)
    
    def run_pipeline(self, docs_dir="Data", query=None):
        print("Processing healthcare documents...")
        chunks = self.load_docs(docs_dir)
        
        if not chunks:
            print("No documents processed. Exiting.")
            return
        
        print(f"Generated {len(chunks)} document segments")
        print("Building searchable knowledge vault...")
        self.build_vault(chunks)
        print("Knowledge vault ready")
        
        test_query = query or "What are the symptoms of cardiovascular disease?"
        print(f"Testing retrieval with query: '{test_query}'")
        
        results = self.search(test_query)
        self.display(results)
        print("\nPipeline completed successfully!")

def main():
    gdrive_creds_path = os.getenv("GDRIVE_CREDENTIALS_PATH")
    hub = HealthcareHub(gdrive_creds_path=gdrive_creds_path)
    hub.run_pipeline(docs_dir="Data", query="What is hypertension treatment?")

if __name__ == "__main__":
    main()
