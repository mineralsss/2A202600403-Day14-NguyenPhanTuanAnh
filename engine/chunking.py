import os
import chromadb
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def read_file_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def semantic_chunk(text, llm, max_chunk_size=500):
    """
    Use OpenAI 4o mini to split text into semantic chunks.
    """
    prompt = (
        "Chia đoạn văn sau thành các đoạn nhỏ hợp lý về mặt ngữ nghĩa, mỗi đoạn không quá "
        f"{max_chunk_size} ký tự. Trả về mỗi đoạn trên một dòng.\n\n{text}"
    )
    response = llm.invoke(prompt)
    if hasattr(response, 'content'):
        return [chunk.strip() for chunk in response.content.split('\n') if chunk.strip()]
    return [chunk.strip() for chunk in str(response).split('\n') if chunk.strip()]

def save_chunks_to_chromadb(chunks, collection_name="doc_chunks", source=None):
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.chromadb"))
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    for i, chunk in enumerate(chunks):
        meta = {"source": source} if source else {}
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}_{os.path.basename(source) if source else 'unknown'}"],
            metadatas=[meta]
        )
    print(f"Saved {len(chunks)} chunks to ChromaDB collection '{collection_name}' from {source}")

def process_and_chunk_documents(doc_paths):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
    for doc_path in doc_paths:
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), doc_path))
        text = read_file_content(abs_path)
        chunks = semantic_chunk(text, llm)
        save_chunks_to_chromadb(chunks, collection_name="doc_chunks", source=doc_path)

if __name__ == "__main__":
    doc_paths = [
        "../data/HARD_CASES_GUIDE.md",
        "../GRADING_RUBRIC.md",
        "../README.md"
    ]
    process_and_chunk_documents(doc_paths)
