from pathlib import Path
import uuid

import chromadb
import ollama
from pypdf import PdfReader


DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


def read_txt(file_path):
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        pages = []

        print("PDF pages:", len(reader.pages))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                pages.append(text)
            else:
                print(f"Page {page_number}: no text extracted")

        return "\n".join(pages)

    except Exception as e:
        print("Error reading PDF:", e)
        return ""


def load_documents():
    documents = []

    for file_path in Path(DOCUMENTS_DIR).glob("*"):
        if file_path.suffix.lower() == ".txt":
            text = read_txt(file_path)
        elif file_path.suffix.lower() == ".pdf":
            text = read_pdf(str(file_path))
        else:
            continue

        if text.strip():
            documents.append({
                "source": str(file_path),
                "text": text
            })

    return documents


def split_text(text, chunk_size=1000, overlap=200):
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


def index_documents():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    documents = load_documents()

    if not documents:
        print("No readable .txt or .pdf documents found in the documents folder.")
        return

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

    print(f"Indexed {total_chunks} chunks successfully.")


def retrieve_context(question, top_k=10):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

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


def ask_question(question):
    context, sources = retrieve_context(question)

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the document context below.
If the answer is not found in the context, say:
"I could not find the answer in the provided documents."

Document context:
{context}

Question:
{question}

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

    print("\nAnswer:")
    print(response["message"]["content"])

    print("\nSources used:")
    shown_sources = set()
    for source in sources:
        if source["source"] not in shown_sources:
            print("-", source["source"])
            shown_sources.add(source["source"])


def main():
    print("Simple Ollama RAG System")
    print("1. Index documents")
    print("2. Ask questions")
    print("3. Exit")

    while True:
        choice = input("\nChoose an option: ")

        if choice == "1":
            index_documents()

        elif choice == "2":
            question = input("Ask your question: ")
            ask_question(question)

        elif choice == "3":
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()