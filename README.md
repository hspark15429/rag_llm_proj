# AI-Powered Document Q&A System

This application is an AI-powered document question-answering system that uses natural language processing to provide answers based on uploaded PDF documents. It leverages Flask for the backend, ChromaDB for vector storage, and various LangChain components for text processing and retrieval.

## Features

- PDF document upload and processing
- AI-powered question answering based on document content
- Chat history storage and retrieval
- Multiple text splitting strategies for document chunking
- Latency measurement for performance monitoring

## Requirements

- Python 3.12+
- Flask
- LangChain
- ChromaDB
- Ollama
- FastEmbed
- PDFPlumber
- SQLite3

## Installation
1. Download and run Ollama: [https://ollama.com/download](https://ollama.com/download)
2. Pull llama3 model from the terminal
```
ollama pull llama3
1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```
python app.py
```
2. Upload a PDF document:
- Send a POST request to `/upload_pdf` with the PDF file

3. Ask questions:
- Send a POST request to `/chat` with a JSON body containing the query:
  ```json
  {
    "query": "Your question here"
  }
  ```

4. Retrieve chat history:
- Send a GET request to `/get_chat_history`

## Configuration

- The application uses the "tinyllama" model from Ollama. Ensure Ollama is installed and the model is available.
- The vector store and database files are stored in the `db` directory.
- Various text splitting strategies are available. Modify the `text_splitter` variable in `app.py`'s upload_pdf() to experiment with different chunking methods.