# rag_gui_voice.py
# Conversational RAG with Mistral (Ollama) + Hardcoded PDF + Dynamic Suggestions + Voice

import os
import time

import gradio as gr
from gtts import gTTS
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


def suggest_questions_from_text(text, n=5):
    """Generate sample student questions from text"""
    llm = Ollama(model="mistral:7b-instruct")
    prompt = f"""
    You are a helpful tutor. Based on the following text, generate {n} insightful
    questions a student might ask to understand it better. 
    Return each question on a new line.

    {text[:1200]}
    """
    result = llm(prompt)
    return [q.strip("-• ") for q in result.strip().split("\n") if q.strip()]


def text_to_speech(answer_text, filename="answer.mp3"):
    """Convert answer to speech using gTTS"""
    tts = gTTS(answer_text, lang="en", tld="co.uk")
    tts.save(filename)
    return filename


# ---------- Build System ----------
print("🔎 Building RAG system... (first run may take a bit)")
qa, chunks = build_rag(PDF_FILE)
print("✅ Ready! Ask questions about", PDF_FILE)

# Initial suggestions
suggested_qs = suggest_questions_from_text(chunks[0].page_content, n=5)


# ---------- Gradio GUI ----------
with gr.Blocks() as demo:
    gr.Markdown(
        "# 📘 RAG Tutor (Mistral + PDF + Voice)\nChat with your textbook chapter"
    )
    gr.Markdown("**Model:** mistral:7b-instruct")

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask a question")
    status = gr.Label(value="Idle ✅")
    audio_output = gr.Audio(label="🔊 Answer Voice", type="filepath")

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
        yield history, "⏳ AI is thinking...", None

        answer = qa.run(user_msg)
        elapsed = time.time() - start
        history[-1] = (user_msg, answer)
        audio_file = text_to_speech(answer)  # generate speech
        yield history, f"✅ Answered in {elapsed:.2f} sec", audio_file

    def refresh_suggestions(history):
        if history and history[-1][1]:
            last_answer = history[-1][1]
            new_qs = suggest_questions_from_text(last_answer, n=5)
        else:
            new_qs = suggested_qs
        return [gr.update(value=q) for q in new_qs[:5]]

    msg.submit(user_message, [msg, state], [msg, state]).then(
        bot_message, state, [chatbot, status, audio_output]
    ).then(refresh_suggestions, state, suggestion_btns)

    for btn in suggestion_btns:
        btn.click(lambda x=btn.value: (x, []), outputs=[msg, state]).then(
            user_message, [msg, state], [msg, state]
        ).then(bot_message, state, [chatbot, status, audio_output]).then(
            refresh_suggestions, state, suggestion_btns
        )

demo.launch()
