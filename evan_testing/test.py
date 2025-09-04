import os

import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")

output_folder = "voice_samples"
os.makedirs(output_folder, exist_ok=True)

for i, voice in enumerate(voices):
    engine.setProperty("voice", voice.id)
    text = f"Hello, this is a test of the selected voice. code{i}"
    filename = os.path.join(output_folder, f"voice_{i}.mp3")

    engine.save_to_file(text, filename)
    engine.runAndWait()  # Run immediately for this voice
    print(f"Saved: {filename}")

print("All voice samples generated successfully.")
