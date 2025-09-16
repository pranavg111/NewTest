pip install pypdf

from flask import Flask, request, jsonify
import os
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# -------------------
# Load and process PDF
# -------------------
loader = PyPDFLoader("Pranav_Resume.pdf")
pages = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ---------------
# Flask App
# ---------------
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("query", "")
    result = qa_chain.invoke({"query": query})
    return jsonify({"question": query, "answer": result["result"]})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0")

# app = Flask(__name__)

# @app.route("/ask", methods=["POST"])
# def ask_question():
#     data = request.get_json()
#     query = data.get("query", "")
#     # Always return the same answer
#     return jsonify({"question": query, "answer": "hello world"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0")







