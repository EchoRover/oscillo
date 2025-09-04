# rag_gui.py
# Conversational RAG with Mistral (Ollama) + Hardcoded PDF + Suggested Questions

import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

PDF_FILE = "chapter.pdf"  # 👉 Hardcoded textbook chapter


def build_rag(pdf_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(chunks, embeddings)

    # 4. LLM
    llm = Ollama(model="mistral:7b-instruct")

    # 5. Conversational chain with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(), memory=memory
    )

    return qa, chunks


def suggest_questions(chunks, n=5):
    """Generate sample student questions from first chunk"""
    text = chunks[0].page_content[:1500]  # just first chunk
    llm = Ollama(model="mistral:7b-instruct")
    prompt = f"""
    You are a helpful tutor. Based on the following text, generate {n} insightful
    questions a student might ask to understand it better:

    {text}
    """
    result = llm(prompt)
    return result.strip().split("\n")


# ---------- Build System ----------
print("🔎 Building RAG system... (first run may take a bit)")
qa, chunks = build_rag(PDF_FILE)
print("✅ Ready! Ask questions about", PDF_FILE)

# Pre-generate questions
suggested_qs = suggest_questions(chunks, n=5)


# ---------- Gradio GUI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 📘 RAG Tutor (Mistral + PDF)\nChat with your textbook chapter")

    # Chat UI
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask a question")

    # Suggested questions
    with gr.Row():
        suggestion_buttons = [gr.Button(q) for q in suggested_qs[:5]]

    state = gr.State([])  # to hold chat history

    def user_message(user_msg, history):
        history = history + [(user_msg, None)]
        return "", history

    def bot_message(history):
        user_msg = history[-1][0]
        answer = qa.run(user_msg)
        history[-1] = (user_msg, answer)
        return history

    msg.submit(user_message, [msg, state], [msg, state]).then(
        bot_message, state, state
    ).then(lambda h: h, state, chatbot)

    for btn in suggestion_buttons:
        btn.click(lambda x=btn.value: (x, []), outputs=[msg, state])

demo.launch()
