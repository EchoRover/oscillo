import gradio as gr
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


def build_rag(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(chunks, embeddings)

    llm = Ollama(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa


# Preload the PDF
qa = build_rag("chapter.pdf")


def answer_question(query):
    return qa.run(query)


# Simple Gradio UI
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the textbook..."),
    outputs="text",
    title="📘 Textbook RAG Reader",
    description="Ask any question from the PDF, answered by Mistral + Retrieval-Augmented Generation.",
)

if __name__ == "__main__":
    demo.launch()
