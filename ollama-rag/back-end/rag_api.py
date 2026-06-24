from pathlib import Path
import uuid
import shutil

import chromadb
import ollama
from pypdf import PdfReader

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
    question: str


def read_txt(file_path):
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def load_documents():
    documents = []

    Path(DOCUMENTS_DIR).mkdir(exist_ok=True)

    for file_path in Path(DOCUMENTS_DIR).glob("*"):
        suffix = file_path.suffix.lower()
        text = ""

        if suffix == ".txt":
            text = read_txt(file_path)

        elif suffix == ".pdf":
            text = read_pdf(str(file_path))

        else:
            continue

        if text.strip():
            documents.append({
                "source": str(file_path),
                "text": text
            })

    return documents


def split_text(text, chunk_size=2000, overlap=200):
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def embed_text(text):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )
    return response["embeddings"][0]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


@app.get("/")
def home():
    return {"message": "Ollama RAG API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    Path(DOCUMENTS_DIR).mkdir(exist_ok=True)

    file_path = Path(DOCUMENTS_DIR) / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename
    }


@app.post("/index")
def index_documents():
    collection = get_collection()
    documents = load_documents()

    if not documents:
        return {
            "message": "No readable .txt or text-based .pdf documents found.",
            "chunks_indexed": 0
        }

    total_chunks = 0

    for doc in documents:
        chunks = split_text(doc["text"])

        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": doc["source"],
                    "chunk": i
                }]
            )

            total_chunks += 1

    return {
        "message": "Documents indexed successfully",
        "chunks_indexed": total_chunks
    }


def retrieve_context(question, top_k=5):
    collection = get_collection()

    question_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context = ""

    for chunk, source in zip(chunks, sources):
        context += f"\nSource: {source['source']}\n"
        context += chunk
        context += "\n---\n"

    return context, sources


@app.post("/ask")
def ask_question(request: QuestionRequest):
    context, sources = retrieve_context(request.question)

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the document context below.
If the answer is not found in the context, say:
"I could not find the answer in the provided documents."

Document context:
{context}

Question:
{request.question}

Answer:
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    unique_sources = []
    seen = set()

    for source in sources:
        source_name = source["source"]
        if source_name not in seen:
            unique_sources.append(source_name)
            seen.add(source_name)

    return {
        "answer": response["message"]["content"],
        "sources": unique_sources
    }