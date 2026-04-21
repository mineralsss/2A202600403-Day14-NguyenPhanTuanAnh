import asyncio
from typing import List, Dict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()


import os
import chromadb

def load_chunks_from_chromadb(collection_name="doc_chunks"):
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.chromadb"))
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    results = collection.get()
    docs = []
    for doc, meta in zip(results["documents"], results.get("metadatas", [{}]*len(results["documents"]))):
        docs.append(Document(page_content=doc, metadata=meta))
    return docs


class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    """
    def __init__(self, top_k: int = 2, system_prompt: str = None,
                 model_name: str = "gpt-4o-mini", openrouter: bool = False):
        self.name = "SupportAgent"
        self.top_k = top_k
        self.model_name = model_name
        self.system_prompt = system_prompt or (
            "Bạn là trợ lý AI. Hãy trả lời câu hỏi sau dựa trên các đoạn tài liệu được cung cấp. "
            "Nếu không đủ thông tin, hãy nói rõ."
        )
        if openrouter:
            self.llm = ChatOpenAI(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        else:
            self.llm = ChatOpenAI(model=model_name)
        # Load chunks from ChromaDB
        self.docs = load_chunks_from_chromadb()
        self.doc_texts = [doc.page_content for doc in self.docs]
        # BM25
        self.bm25 = BM25Okapi([doc.page_content.lower().split() for doc in self.docs])
        # Embedding model: OpenAI text-embedding-3-small
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        self.doc_embeddings = np.array(self.embedder.embed_documents(self.doc_texts))

    async def query(self, question: str) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Hybrid BM25 + Vector search.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """
        # 1. Hybrid Retrieval
        # BM25
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Vector search
        query_embedding = np.array(self.embedder.embed_query(question)).reshape(1, -1)
        vector_scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        # Hybrid: weighted sum (equal weight)
        hybrid_scores = (np.array(bm25_scores) + vector_scores) / 2
        top_indices = np.argsort(hybrid_scores)[::-1][:self.top_k]
        matched_contexts = [self.docs[i].page_content[:300] + ("..." if len(self.docs[i].page_content) > 300 else "") for i in top_indices]
        matched_sources = [self.docs[i].metadata.get("source", "unknown") for i in top_indices]

        # 2. Generation: Real LLM call
        prompt = (
            f"{self.system_prompt}\n"
            f"Câu hỏi: {question}\n"
            "Các đoạn tài liệu liên quan:\n"
        )
        for i, ctx in enumerate(matched_contexts):
            prompt += f"[Doc {i+1}]: {ctx}\n"
        response = await self.llm.ainvoke(prompt)
        return {
            "answer": response.content if hasattr(response, 'content') else str(response),
            "contexts": matched_contexts,
            "metadata": {
                "model": self.model_name,
                "token": len(response.content.split()) if hasattr(response, 'content') else len(str(response).split()),
                "sources": matched_sources
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Cách làm bài lab này?")
        print(resp)
    asyncio.run(test())
