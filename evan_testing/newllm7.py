import threading
import time
import gradio as gr
import keyboard
import pyttsx3
import speech_recognition as sr
from collections import deque

# ---------- TTS ----------
engine = pyttsx3.init()
voices = engine.getProperty("voices")
for v in voices:
    if "male" in v.name.lower() or "english" in v.name.lower():
        engine.setProperty("voice", v.id)
        break
engine.setProperty("rate", 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------- Voice ----------
recognizer = sr.Recognizer()
voice_queue = deque()

def listen_once():
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text
    except Exception as e:
        print("⚠️ Voice error:", e)
        return None

def background_listener():
    while True:
        keyboard.wait("v")
        q = listen_once()
        if q:
            voice_queue.append(q)

# ---------- Gradio ----------
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Voice Tutor")
    gr.Markdown("Press **V** and speak...")

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask a question")
    voice_input = gr.Textbox(visible=False)  # hidden box for voice
    state = gr.State([])

    def user_message(user_msg, history):
        return "", history + [(user_msg, None)]

    def bot_message(history):
        user_msg = history[-1][0]
        start = time.time()
        yield history, "⏳ AI is thinking..."
        # Here call your RAG qa.run(user_msg)
        answer = f"Echo: {user_msg}"
        elapsed = time.time() - start
        history[-1] = (user_msg, answer)
        speak(answer)
        yield history, f"✅ Answered in {elapsed:.2f} sec"

    # Submit from text or voice box
    for box in [msg, voice_input]:
        box.submit(user_message, [box, state], [box, state]).then(
            bot_message, state, [chatbot, msg]
        )

    # Periodic poller to inject voice into textbox
    def poll_voice():
        if voice_queue:
            return gr.Textbox.update(value=voice_queue.popleft())
        return gr.Textbox.update()

    demo.load(poll_voice, None, voice_input, every=0.5)

# Start background thread
threading.Thread(target=background_listener, daemon=True).start()

demo.launch()
