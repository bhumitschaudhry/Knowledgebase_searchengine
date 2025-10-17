# Knowledge-Base Search Engine

A lightweight, RAG-powered search system that lets you upload documents and ask natural-language questions.
Instead of matching keywords, it understands your query, retrieves relevant sections, and generates concise answers using an LLM.

## Features

- Upload and process PDF or text files
- Store document chunks as vector embeddings for semantic search
- Retrieve and synthesize answers using RAG (Retrieval-Augmented Generation)
- FastAPI backend with a simple web interface

## How It Works

1. Documents are split into chunks and embedded as vectors.
2. Embeddings are stored in a **FAISS** vector database.
3. The system retrieves the most relevant chunks for each query.
4. A connected LLM (Google Gemini or HuggingFace) generates the final answer.

## Setup

```bash
git clone <repository-url>
cd knowledge-base-search-engine
pip install -r requirements.txt
cp .env.example .env
```

Add your API key in `.env`:

```
GOOGLE_API_KEY=your_key_here
# or
HUGGINGFACE_API_TOKEN=your_token_here
```

Start the app:

```bash
python app.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## API Endpoints

- **POST /upload** — Upload PDF or text files
- **POST /query** — Ask a question and get an AI-generated answer

## Future Plans

- Support for more file types
- Scalable vector database
- User authentication and feedback system
