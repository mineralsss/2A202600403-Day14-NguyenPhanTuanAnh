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
    Hỗ trợ Qwen3.5-4B-Base pipeline khi use_qwen=True.
    """
    def __init__(self, top_k: int = 2, system_prompt: str = None,
                 model_name: str = "gpt-4o-mini", openrouter: bool = False,
                 use_qwen: bool = False):
        self.name = "SupportAgent"
        self.top_k = top_k
        self.model_name = model_name
        self.use_qwen = use_qwen
        self.system_prompt = system_prompt or (
            "Bạn là trợ lý AI. Hãy trả lời câu hỏi sau dựa trên các đoạn tài liệu được cung cấp. "
            "Nếu không đủ thông tin, hãy nói rõ."
        )

        # --- Qwen pipeline mode ---
        if use_qwen:
            from transformers import pipeline as hf_pipeline
            from sentence_transformers import SentenceTransformer

            print("🔄 Loading Qwen3.5-4B-Base pipeline...")
            self.qwen_pipe = hf_pipeline(
                "text-generation",
                model="unsloth/Qwen3.5-4B-Base",
            )
            print("✅ Qwen pipeline loaded.")

            # Use a Qwen-compatible embedding model
            print("🔄 Loading Qwen embedding model...")
            self.qwen_embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
            print("✅ Qwen embedding model loaded.")

            self.llm = None  # Not used in Qwen mode
        else:
            self.qwen_pipe = None
            self.qwen_embedder = None
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

        # Embedding model: Qwen or OpenAI
        if use_qwen:
            self.embedder = None  # use self.qwen_embedder instead
            self.doc_embeddings = np.array(
                self.qwen_embedder.encode(self.doc_texts, show_progress_bar=True)
            )
        else:
            self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
            self.doc_embeddings = np.array(self.embedder.embed_documents(self.doc_texts))

    def _qwen_generate(self, prompt: str) -> str:
        """Generate text using the Qwen text-generation pipeline."""
        result = self.qwen_pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,  # Only return the generated part, not the prompt
        )
        return result[0]["generated_text"].strip()

    async def query(self, question: str) -> Dict:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Hybrid BM25 + Vector search.
        2. Generation: Gọi LLM (Qwen hoặc OpenAI) để sinh câu trả lời.
        """
        # 1. Hybrid Retrieval
        # BM25
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Vector search
        if self.use_qwen:
            query_embedding = self.qwen_embedder.encode([question])
            query_embedding = np.array(query_embedding).reshape(1, -1)
        else:
            query_embedding = np.array(self.embedder.embed_query(question)).reshape(1, -1)

        vector_scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Hybrid: weighted sum (equal weight)
        hybrid_scores = (np.array(bm25_scores) + vector_scores) / 2
        top_indices = np.argsort(hybrid_scores)[::-1][:self.top_k]
        matched_contexts = [self.docs[i].page_content[:300] + ("..." if len(self.docs[i].page_content) > 300 else "") for i in top_indices]
        matched_sources = [self.docs[i].metadata.get("source", "unknown") for i in top_indices]

        # 2. Generation
        prompt = (
            f"{self.system_prompt}\n"
            f"Câu hỏi: {question}\n"
            "Các đoạn tài liệu liên quan:\n"
        )
        for i, ctx in enumerate(matched_contexts):
            prompt += f"[Doc {i+1}]: {ctx}\n"

        if self.use_qwen:
            # Run Qwen pipeline in thread to not block async loop
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(None, self._qwen_generate, prompt)
        else:
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer": answer,
            "contexts": matched_contexts,
            "metadata": {
                "model": "Qwen3.5-4B-Base" if self.use_qwen else self.model_name,
                "token": len(answer.split()),
                "sources": matched_sources
            }
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Cách làm bài lab này?")
        print(resp)
    asyncio.run(test())
