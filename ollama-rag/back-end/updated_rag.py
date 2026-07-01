from pathlib import Path
import uuid

import chromadb
import ollama


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


def load_documents():
    documents = []

    for file_path in Path(DOCUMENTS_DIR).glob("*"):
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext == ".txt" or ext == ".md":
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
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )
    return response["embeddings"][0]


def index_documents():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Delete old collection if it exists, start fresh
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Cleared old index.")
    except:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    chunks = split_documents(documents)
    total = 0

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk.page_content)

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk": i
            }]
        )
        total += 1

    print(f"Indexed {total} chunks successfully.")


def retrieve_context(question, top_k=5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

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
        if distance < 1.0:  # only include relevant chunks
            context += f"\nSource: {source['source']}\n"
            context += chunk
            context += "\n---\n"
            filtered_sources.append(source)

    return context, filtered_sources

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
        source_key = f"{source['source']} (page {source.get('page', 0) + 1})"
        if source_key not in shown_sources:
            print("-", source_key)
            shown_sources.add(source_key)


def main():
    print("Simple Ollama RAG System")
    print("Indexing documents...")
    index_documents()

    while True:
        question = input("\nAsk a question (or 'quit' or 'q' to exit): ")

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question.strip():
            continue

        ask_question(question)


if __name__ == "__main__":
    main()