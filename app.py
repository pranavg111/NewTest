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
# import os
# import logging
# from flask import Flask, request, jsonify

# # LangChain imports
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.vectorstores import InMemoryVectorStore
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.chains import RetrievalQA

# # Logging
# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger("resume-rag")

# app = Flask(__name__)

# # Global lazy-initialized chain
# qa_chain = None

# def find_resume_path():
#     """Locate resume PDF in likely locations."""
#     candidates = [
#         "Pranav_Resume.pdf",
#         os.path.join(os.getcwd(), "Pranav_Resume.pdf"),
#         "/home/site/wwwroot/Pranav_Resume.pdf",  # Azure wwwroot
#     ]
#     for p in candidates:
#         if p and os.path.exists(p):
#             return p
#     return None

# def load_documents_from_pdf(path):
#     """Load and split PDF into LangChain documents."""
#     loader = PyPDFLoader(path)
#     pages = loader.load()
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     return splitter.split_documents(pages)

# def load_documents_from_text(text):
#     """Create documents from plain text."""
#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     return splitter.create_documents([text])

# def init_chain():
#     """Lazily initialize embeddings, InMemoryVectorStore, LLM, and RetrievalQA chain."""
#     global qa_chain
#     if qa_chain is not None:
#         return

#     log.info("Initializing RAG chain (this may take a few seconds)...")

#     # 1) Load resume docs (PDF if found, else fallback text)
#     resume_path = find_resume_path()
#     if resume_path:
#         try:
#             log.info(f"Loading resume from PDF: {resume_path}")
#             docs = load_documents_from_pdf(resume_path)
#         except Exception as e:
#             log.exception("Failed to load PDF; falling back to inline text.")
#             docs = load_documents_from_text("This is the resume content.")
#     else:
#         log.warning("Pranav_Resume.pdf not found. Using inline resume text instead.")
#         docs = load_documents_from_text("This is the resume content.")

#     # 2) Embeddings + LLM
#     openai_key = os.getenv("OPENAI_API_KEY")
#     if not openai_key:
#         log.error("OPENAI_API_KEY not set in environment. The chain cannot be initialized.")
#         raise RuntimeError("OPENAI_API_KEY not set")

#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-ada-002",
#         openai_api_key=openai_key
#     )
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0,
#         openai_api_key=openai_key
#     )

#     # 3) InMemoryVectorStore (safe for Azure, no sqlite issues)
#     try:
#         log.info("Creating InMemoryVectorStore ...")
#         vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
#     except Exception:
#         log.exception("Failed to create InMemoryVectorStore.")
#         raise

#     # 4) RetrievalQA chain
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff"
#     )

#     log.info("RAG chain initialized successfully.")

# @app.route("/")
# def home():
#     return "✅ App is running with RAG!"

# @app.route("/ask", methods=["POST"])
# def ask_question():
#     init_chain()  # ensure models are ready
#     data = request.get_json()
#     query = data.get("query", "")
#     result = qa_chain.invoke({"query": query})
#     return jsonify({"question": query, "answer": result["result"]})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)

###################################################################
import os
import logging
from flask import Flask, request, jsonify
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resume-rag")

app = Flask(__name__)

qa_chain = None

def init_chain():
    global qa_chain
    if qa_chain is not None:
        return

    log.info("Initializing RAG chain...")

    # Resume content as plain text
    resume_text = "This is the resume content."
    docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([resume_text])

    # OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # Embeddings and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    # Instead of vectorstore, we can do simple retrieval by chunk matching
    from langchain.chains import LLMChain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=None, chain_type="stuff")
    qa_chain.docs = docs  # store chunks directly

@app.route("/")
def home():
    return "✅ App is running!"

@app.route("/ask", methods=["POST"])
def ask_question():
    init_chain()
    data = request.get_json()
    query = data.get("query", "")

    # Simple retrieval: find the chunk with the query in it
    result_text = ""
    for doc in qa_chain.docs:
        if query.lower() in doc.page_content.lower():
            result_text = doc.page_content
            break
    if not result_text:
        result_text = "I couldn't find a direct answer in the resume."

    # Call LLM on retrieved chunk
    response = qa_chain.llm.invoke(result_text)
    return jsonify({"question": query, "answer": response.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)



















