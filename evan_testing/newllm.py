# rag_reader.py
# Simple RAG pipeline with Mistral (Ollama) + textbook PDF

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


def build_rag(pdf_path: str):
    # 1. Load textbook chapter
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split into chunks (to fit context window)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Create embeddings + vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(chunks, embeddings)

    # 4. Local LLM (Mistral via Ollama)
    llm = Ollama(model="mistral")

    # 5. RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa


if __name__ == "__main__":
    # 👉 Change this to your textbook chapter PDF
    pdf_file = "chapter.pdf"

    print("🔎 Building RAG system... (first run may take a bit)")
    qa = build_rag(pdf_file)
    print("✅ Ready! Ask questions about", pdf_file)

    while True:
        query = input("\n❓ Question (or 'exit'): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        answer = qa.run(query)
        print("\n📘 Answer:", answer)
