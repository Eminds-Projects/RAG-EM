\# Local RAG System with Ollama



This is a simple local Retrieval-Augmented Generation system built with Python and Ollama.



\## Features



\- Loads documents from a `documents/` folder

\- Supports `.txt` and text-based `.pdf` files

\- Splits documents into chunks

\- Uses `nomic-embed-text` for embeddings

\- Stores document embeddings in ChromaDB

\- Uses `llama3.2:3b` to answer questions from retrieved document context



\## Requirements



\- Python 3.10+

\- Ollama installed locally



\## Ollama Models



Pull the required models:



```bash

ollama pull llama3.2:3b

ollama pull nomic-embed-text



SETUP 



Create a virtual environment 

python -m venv venv



Install dependencies

pip install -r requirements.txt



python rag.py to run

