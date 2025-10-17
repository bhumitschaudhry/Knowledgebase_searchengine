import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
VECTOR_STORE_PATH = "vector_store.pkl"
def get_vector_store():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            vector_store = pickle.load(f)
        vector_store._embedding_function = embeddings
    else:
        vector_store = FAISS(embedding_function=embeddings, allow_dangerous_deserialization=True)
    return vector_store
def save_vector_store(vector_store):
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)