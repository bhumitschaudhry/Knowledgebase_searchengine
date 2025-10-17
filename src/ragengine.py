import os
from typing import List, Dict, Tuple, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFaceHub
from src.vector_store import get_vector_store, save_vector_store
class RAGEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        huggingface_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        if google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=google_api_key,
                temperature=0.1
            )
        elif huggingface_api_key:
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=huggingface_api_key
            )
        else:
            raise ValueError("No API key found for language models. Please set GOOGLE_API_KEY or HUGGINGFACE_API_TOKEN in .env file.")
        self.prompt_template = """
        You are a helpful assistant that provides accurate information based on the given context.
        Context:
        {context}
        Question: {question}
        Please provide a clear, concise, and accurate answer based only on the information in the context.
        If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
        Answer:
        """
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    def query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        result = self.qa_chain({"query": query})
        answer = result["result"]
        source_docs = result["source_documents"]
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "doc_id": doc.metadata.get("doc_id", "Unknown")
            })
        return answer, sources
    def add_documents(self, documents):
        self.vector_store.add_documents(documents)
        save_vector_store(self.vector_store)