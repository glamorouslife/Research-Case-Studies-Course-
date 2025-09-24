# MediRAG System

A medical Q\&A assistant powered by **Retrieval-Augmented Generation (RAG)** with **BioMistral-7B** and a **Qdrant vector database**.

## Features

* **GPU/CPU Support** → CUDA acceleration with CPU fallback
* **Medical LLM** → BioMistral-7B, fine-tuned for the medical domain
* **Vector Database** → Qdrant for fast and accurate retrieval
* **PubMed Embeddings** → Domain-specific embeddings for higher accuracy
* **Web Interface** → Simple FastAPI app with a browser UI
* **Evaluation Tools** → Compare raw LLM answers vs RAG answers

---

## Core Concepts

### What is Retrieval-Augmented Generation (RAG)?

RAG makes an LLM smarter by letting it “look things up.”
Instead of depending only on its old training data, the model retrieves fresh, relevant documents from a knowledge base (here: Qdrant) and uses them to answer questions more reliably.

How it works in this project:

1. You ask a medical question.
2. The system looks in Qdrant for the most relevant documents.
3. Those documents + your question are given to BioMistral-7B.
4. The model generates an informed answer, backed by real sources.

### What is “Ground Truth”?

In this setup, *ground truth* means the **verified medical documents** we’ve ingested (e.g., guidelines, clinical handbooks).
The system’s answers are grounded in these documents → meaning you can always trace the response back to a reliable source clinically proven information.

This matters for medicine: you don’t want “AI magic,” you want **transparent, verifiable, trustworthy answers**.

---

## Quick Start

### Prerequisites

* Python 3.10+
* Docker Desktop (with Docker running)
* NVIDIA GPU (optional, for faster inference)

---

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download BioMistral model

* Get `BioMistral-7B.Q4_K_S.gguf` from [BioMistral Collection](https://huggingface.co/collections/BioMistral)
* Place it in the **project root folder**

---

### 3. Set up Google Drive API credentials (optional for ingestion)

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → create a new project.
2. Enable **Google Drive API**.
3. Create OAuth credentials (Desktop app).
4. Download the JSON file → rename it `credentials.json`.
5. Move it to the `src` folder.
6. Create a `.env` file inside `src` with:
## Keeping Your Knowledge Base Fresh (Google Drive & Local)

- **Google Drive:**  
  When you add, edit, or replace documents in your linked Drive folder, re-run the ingestion (`qd_retrive_new.py`).  
  This updates the vector index so the system uses the latest medical information.

- **Tip:**  
  Keep filenames stable so updates are treated as replacements rather than duplicates.

   ```bash
   GDRIVE_CREDENTIALS_PATH="./src/credentials.json"
   ```

---

### 4. Start Qdrant (Vector Database)

Here’s the friendly step-by-step:

1. Install & open **Docker Desktop**.
2. Open **PowerShell**.
3. Pull Qdrant:

   ```bash
   docker pull qdrant/qdrant
   ```
4. See your images:

   ```bash
   docker images
   ```
5. See running containers:

   ```bash
   docker ps
   ```
6. Run Qdrant:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

---

### 5. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

* **Windows (PowerShell):**

  ```bash
  .venv\Scripts\activate
  ```
* **Linux/Mac:**

  ```bash
  source .venv/bin/activate
  ```

---

### 6. Ingest documents + test

* To process local PDFs in the `Data` folder and run a query:

  ```bash
  python qd_retrive_new.py
  ```
* To pull docs directly from Google Drive, pass the folder ID in `run_pipeline()` inside `qd_retrive_new.py`.

---

### 7. Start the Web App

```bash
python rag_main.py
```

→ Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

### 8. Run Evaluation (Optional)

Compare RAG vs. plain LLM answers:

```bash
python generate_evaluation_queries.py
```

---

## Configuration

### GPU settings

* 4GB → `gpu_layers=15`
* 8GB → `gpu_layers=25`
* 12GB+ → `gpu_layers=32+`

### CPU only

* Set `lib="avx2"`
* Remove `gpu_layers`

---

## API Endpoints

* `GET /` → Web UI
* `POST /get_response` → Submit a query
* `GET /health` → Check system status

**Response format**:

```json
{
  "answer": "Medical response based on context",
  "source_document": "Retrieved medical literature",
  "doc": "Source metadata"
}
```

---

## File Structure

```
├── rag_main.py                     # FastAPI server
├── qd_retrive_new.py               # Ingestion + pipeline
├── generate_evaluation_queries.py  # Evaluation script
├── requirements.txt
├── .env
├── .gitignore
├── Data/                           # Store local PDFs here
├── templates/                      # Web UI templates
├── docker.txt
├── medical_rag_evaluation_5.json   # Example evaluation results
└── README.md
```

---

## Troubleshooting

* **CUDA errors** → use CPU mode (`lib="avx2"`)
* **Memory issues** → lower `gpu_layers` or `max_new_tokens`
* **Connection refused** → check Qdrant is running on port `6333`
* **Port already in use** → the server will auto-pick a free port

---

## Performance Tips

* Use GPU if available (much faster)
* Tune `gpu_layers` to match your VRAM
* For CPUs → set `threads = half your cores`
* Check GPU usage with:

  ```bash
  nvidia-smi
  ```
