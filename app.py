# from flask import Flask, request, jsonify
# import os
# #from pydantic import BaseModel
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.chains import RetrievalQA


# text = "This is the resume content."
# splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.create_documents([text])

# embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))


# vectorstore = Chroma.from_documents(docs, embedding_model)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff"
# )


# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "✅ App is running with RAG!"

# @app.route("/ask", methods=["POST"])
# def ask_question():
#     data = request.get_json()
#     query = data.get("query", "")

#     # Use RetrievalQA (resume-aware)
#     result = qa_chain.invoke({"query": query})

#     return jsonify({"question": query, "answer": result["result"]})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0")

#################################################################################

# app.py
import os
import logging
from flask import Flask, request, jsonify

# LangChain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.vectorstores import Chroma
from langchain.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resume-rag")

app = Flask(__name__)

# Global lazy-initialized chain
qa_chain = None

def find_resume_path():
    """
    Look for the resume PDF in a few likely places.
    Returns path if found, else None.
    """
    candidates = [
        "Pranav_Resume.pdf",
        os.path.join(os.getcwd(), "Pranav_Resume.pdf"),
        "/home/site/wwwroot/Pranav_Resume.pdf",  # typical Azure wwwroot
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def load_documents_from_pdf(path):
    """
    Load and split PDF into LangChain documents.
    Raises if loader fails.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    return docs

def load_documents_from_text(text):
    """
    Create documents from plain text (fallback).
    """
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

def init_chain():
    """
    Lazily initialize embeddings, vectorstore (Chroma), LLM, and RetrievalQA chain.
    This runs on first /ask request to avoid worker startup failures in Azure.
    """
    global qa_chain
    if qa_chain is not None:
        return

    log.info("Initializing RAG chain (this may take a few seconds)...")

    # 1) Get resume docs (PDF if present, else fallback text)
    resume_path = find_resume_path()
    if resume_path:
        try:
            log.info(f"Loading resume from PDF: {resume_path}")
            docs = load_documents_from_pdf(resume_path)
        except Exception as e:
            log.exception("Failed to load PDF; falling back to inline text.")
            docs = load_documents_from_text("This is the resume content.")
    else:
        log.warning("Pranav_Resume.pdf not found. Using inline resume text instead.")
        docs = load_documents_from_text("This is the resume content.")

    # 2) Embeddings and LLM (read API key from environment)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        log.error("OPENAI_API_KEY not set in environment. The chain cannot be initialized.")
        raise RuntimeError("OPENAI_API_KEY not set")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_key
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_key
    )

    # 3) Chroma vectorstore (persist in /tmp which is writable on Azure)
    persist_dir = "/tmp/chroma_store"
    try:
        log.info(f"Creating Chroma vectorstore at {persist_dir} ...")
        # vectorstore = Chroma.from_documents(
        #     docs,
        #     embedding=embeddings,
        #     persist_directory=persist_dir
        # )
        vectorstore = InMemoryVectorStore.from_documents(docs, embedding_model)

    except TypeError:
        # Some versions expect different kwarg names — try alternate signature
        log.warning("Chroma.from_documents signature mismatch; trying alternate call.")
        vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_dir
        )
    except Exception:
        log.exception("Failed to create Chroma vectorstore.")
        raise

    # 4) Build RetrievalQA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    log.info("RAG chain initialized successfully.")

@app.route("/")
def home():
    return "✅ App is running with RAG!"

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        init_chain()
    except Exception as e:
        # Initialization failed — return a helpful error and log details
        log.exception("Initialization error")
        return jsonify({"error": "Server initialization failed. Check server logs."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Please POST JSON with a 'query' field."}), 400

    query = data.get("query", "")

    try:
        # Use RetrievalQA to answer grounded in resume content
        result = qa_chain.invoke({"query": query})
        # result may be a dict-like depending on langchain version
        answer = result.get("result") if isinstance(result, dict) else getattr(result, "result", None)
        if answer is None:
            # fallback to other property names
            answer = getattr(result, "content", str(result))
        return jsonify({"question": query, "answer": answer})
    except Exception:
        log.exception("Error while answering query")
        return jsonify({"error": "Failed to generate answer. Check server logs."}), 500

if __name__ == "__main__":
    # Local dev: run on port 8000 to match Azure warmup ping expectations
    app.run(host="0.0.0.0", port=8000)


















