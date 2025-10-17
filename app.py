from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import uvicorn
from dotenv import load_dotenv
from typing import List, Optional
from src.process_doc import process_documents
from src.ragengine import RAGEngine
load_dotenv()
app = FastAPI(title="Knowledge Base Search Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rag_engine = RAGEngine()
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        file_paths = []
        for file in files:
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)
        doc_ids = process_documents(file_paths)
        return {"status": "success", "message": f"Processed {len(doc_ids)} documents", "doc_ids": doc_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/query")
async def query_knowledge_base(query: str = Form(...)):
    try:
        answer, sources = rag_engine.query(query)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)