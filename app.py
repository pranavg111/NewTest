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
from flask import Flask, request, jsonify
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

app = Flask(__name__)

# Lazy init so Azure doesn't fail on boot
llm = None
vectorstore = None

def init_models():
    global llm, vectorstore
    if llm is None or vectorstore is None:
        # 1. Resume content
        text = "This is the resume content."
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])

        # 2. Embeddings + LLM
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # 3. Chroma vectorstore (use /tmp for Azure)
        vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory="/tmp/chroma_store"
        )

@app.route("/")
def home():
    return "✅ App is running!"

@app.route("/ask", methods=["POST"])
def ask_question():
    init_models()  # make sure LLM + Chroma are ready
    data = request.get_json()
    query = data.get("query", "")

    # Retrieve context
    docs = vectorstore.similarity_search(query, k=2)
    context = " ".join([d.page_content for d in docs])

    # Generate answer
    result = llm.invoke(f"Context: {context}\n\nQuestion: {query}")

    return jsonify({"question": query, "answer": result.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)














