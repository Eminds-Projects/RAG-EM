from pathlib import Path
import uuid
import shutil

import chromadb
import ollama

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    message: str


# --- RAG functions (same as updated_rag.py) ---

def load_documents():
    documents = []
    Path(DOCUMENTS_DIR).mkdir(exist_ok=True)

    for file_path in Path(DOCUMENTS_DIR).glob("*"):
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                continue
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def embed_text(text):
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def index_documents():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    documents = load_documents()
    if not documents:
        return 0

    chunks = split_documents(documents)

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk.page_content)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "N/A"),
                "chunk": i
            }]
        )

    return len(chunks)


def retrieve_context(question, top_k=5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        return "", []

    question_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    context = ""
    filtered_sources = []

    for chunk, source, distance in zip(chunks, sources, distances):
        if distance < 1.0:
            context += f"\nSource: {source['source']}\n"
            context += chunk
            context += "\n---\n"
            filtered_sources.append(source)

    return context, filtered_sources


# --- API endpoints ---

@app.get("/api/health")
def home():
    return {"message": "Ollama RAG API is running"}


@app.post("/api/documents/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    Path(DOCUMENTS_DIR).mkdir(exist_ok=True)
    filenames = []

    for file in files:
        file_path = Path(DOCUMENTS_DIR) / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filenames.append(file.filename)

    # Index once after all files are saved
    total = index_documents()

    return {
        "message": f"{len(filenames)} files uploaded and indexed",
        "filenames": filenames,
        "chunks_indexed": total
    }

@app.post("/api/chat/send")
def send(request: QuestionRequest):
    context, sources = retrieve_context(request.message)

    if not context:
        return {
            "answer": "No documents indexed yet. Please upload documents first.",
            "sources": []
        }

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the document context below.
If the answer is not found in the context, say:
"I could not find the answer in the provided documents."

Document context:
{context}

Question:
{request.message}

Answer:
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    unique_sources = []
    seen = set()
    for source in sources:
        page = source.get("page", "N/A")
        if page != "N/A":
            page = page + 1
        source_key = f"{source['source']} (page {page})"
        if source_key not in seen:
            unique_sources.append(source_key)
            seen.add(source_key)

    return {

        "role": "assistant",
        "answer": response["message"]["content"],
        "sources": unique_sources
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)