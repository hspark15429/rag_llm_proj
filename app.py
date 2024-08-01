from flask import Flask, jsonify, request
import time
import shutil
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PDFPlumberLoader
from db import init_db, store_response, get_previous_responses
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter


app = Flask(__name__)

init_db()

llm = Ollama(model="llama3")

persist_directory = "db"

embeddings = FastEmbedEmbeddings()

# Current implementation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    length_function=len,
)

# Experiment with different chunk sizes and overlaps
text_splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    length_function=len,
)

text_splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=40,
    length_function=len,
)

# Character-based splitter
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Token-based splitter
token_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)


template = """
<s> User: [INST] Here is the context: "{context}". Based on this context, please provide detailed information to answer {input}. 
If the information is not available in the provided documents, let me know explicitly by saying "I don't have information on this document.[/INST]"
<s> Assistant:
"""

prompt = PromptTemplate.from_template(template)

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400
    print(f"Received query: {query}")
    try:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Loaded vector store")

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.2,
            },
        )

        documents_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, documents_chain)

        res = retrieval_chain.invoke({"input": query})
        
        end_time = time.time()
        latency = end_time - start_time

        store_response(query, res["answer"], latency)

        return jsonify({"response": res["answer"], "latency": latency})
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    # Delete existing vector store
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Existing vector store at {persist_directory} has been removed.")
    # Recreate the persist_directory
    os.makedirs(persist_directory, exist_ok=True)
    print(f"Created new directory at {persist_directory}")

    f = request.files["file"]
    f.save("data/" + f.filename)
    pdf_loader = PDFPlumberLoader("data/" + f.filename)
    docs = pdf_loader.load_and_split()
    # MODIFY HERE to experiment with different chunking strategies
    doc_chunks = text_splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=doc_chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )

    db.persist()

    return jsonify({
        "status": "Upload was successful",
        "filename": f.filename,
    })

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    try:
        print("getting chat history")
        results = get_previous_responses(5)

        history = [{"query": res[0], "response": res[1], "latency": res[2], "timestamp": res[3]} for res in results]
        return jsonify({"chat_history": history})
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)