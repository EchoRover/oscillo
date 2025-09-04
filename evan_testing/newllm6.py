# rag_gui_pyttsx3.py
# Conversational RAG Tutor with Mistral + PDF + pyttsx3 male voice (offline)

import time

import gradio as gr
import pyttsx3
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

PDF_FILE = "chapter.pdf"  # Hardcoded PDF


# ---------- TTS SETUP ----------
engine = pyttsx3.init()
voices = engine.getProperty("voices")

# Pick a male voice (choose one that sounds deep/professor-like)
for v in voices:
    if "male" in v.name.lower() or "english" in v.name.lower():
        engine.setProperty("voice", v.id)
        break

engine.setProperty("rate", 160)  # speaking speed
engine.setProperty("volume", 1.0)  # max volume


def speak(text):
    engine.say(text)
    engine.runAndWait()


# ---------- RAG SETUP ----------
def build_rag(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(chunks, embeddings)

    llm = Ollama(model="mistral:7b-instruct")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(), memory=memory
    )

    return qa, chunks


def suggest_questions_from_text(text, n=5):
    llm = Ollama(model="mistral:7b-instruct")
    prompt = f"""
    You are a helpful tutor. Based on the following text, generate {n} insightful
    questions a student might ask to understand it better. Return each on a new line.

    {text[:1200]}
    """
    result = llm(prompt)
    return [q.strip("-• ") for q in result.strip().split("\n") if q.strip()]


# ---------- Build system ----------
print("🔎 Building RAG system... (first run may take a bit)")
qa, chunks = build_rag(PDF_FILE)
print("✅ Ready! Ask questions about", PDF_FILE)

suggested_qs = suggest_questions_from_text(chunks[0].page_content, n=5)


# ---------- Gradio GUI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 📘 RAG Tutor (Offline Male Voice)\nChat with your textbook chapter")
    gr.Markdown("**Model:** mistral:7b-instruct")

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask a question")
    status = gr.Label(value="Idle ✅")

    suggestion_box = gr.Column()
    with suggestion_box:
        suggestion_btns = [gr.Button(q) for q in suggested_qs]

    state = gr.State([])

    def user_message(user_msg, history):
        history = history + [(user_msg, None)]
        return "", history

    def bot_message(history):
        user_msg = history[-1][0]
        start = time.time()
        yield history, "⏳ AI is thinking..."

        answer = qa.run(user_msg)
        elapsed = time.time() - start
        history[-1] = (user_msg, answer)

        # Speak answer offline
        speak(answer)

        yield history, f"✅ Answered in {elapsed:.2f} sec"

    def refresh_suggestions(history):
        if history and history[-1][1]:
            last_answer = history[-1][1]
            new_qs = suggest_questions_from_text(last_answer, n=5)
        else:
            new_qs = suggested_qs
        return [gr.update(value=q) for q in new_qs[:5]]

    # Main flow
    msg.submit(user_message, [msg, state], [msg, state]).then(
        bot_message, state, [chatbot, status]
    ).then(refresh_suggestions, state, suggestion_btns)

    # Click suggestion → auto-submit
    for btn in suggestion_btns:
        btn.click(lambda x=btn.value: (x, []), outputs=[msg, state]).then(
            user_message, [msg, state], [msg, state]
        ).then(bot_message, state, [chatbot, status]).then(
            refresh_suggestions, state, suggestion_btns
        )

demo.launch()
