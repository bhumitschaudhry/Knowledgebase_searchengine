from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from src.vector_store import get_vector_store
def process_documents(file_paths: List[str]) -> List[str]:
    documents = []
    doc_ids = []
    for file_path in file_paths:
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path)
                docs = loader.load()
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            doc_id = str(uuid.uuid4())
            for doc in docs:
                doc.metadata["source"] = file_path
                doc.metadata["doc_id"] = doc_id
            documents.extend(docs)
            doc_ids.append(doc_id)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return doc_ids