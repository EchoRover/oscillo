from tensorflow.keras.models import load_model
import numpy as np
import random
import pickle

# Load model
model = load_model("phy_three.h5")
filepath = "filtered_physics.txt"
# Load character mappings
with open("char_mappings.pkl", "rb") as f:
    char_to_index, index_to_char = pickle.load(f)

# Character info
chars = sorted(char_to_index.keys())
seq_length = 60  # must match training

# Load original text for fallback seed

with open(filepath, "r", encoding="utf-8") as f:
    text = ""
    for line in f:
        if line.startswith("<PARA>"):
            line = line.replace("<PARA>", "").replace("<END>", "").strip()
            text += line + " § "  # Add stop token after each paragraph


# Sampling function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature  # avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Text generation from custom input
def generate_text(length, temperature=0.6, start=None):
    if start:
        sentence = start.lower()
        if len(sentence) < seq_length:
            sentence = " " * (seq_length - len(sentence)) + sentence
        elif len(sentence) > seq_length:
            sentence = sentence[-seq_length:]
    else:
        start_index = random.randint(0, len(text) - seq_length - 1)
        sentence = text[start_index : start_index + seq_length]

    generated = sentence
    final_text = ""
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(sentence):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
        if len(generated) > seq_length:
            print(next_char, end="", flush=True)
    print()
    return generated


import sys

# Chat loop!
print("Shakespeare AI Chat | Type 'exit' to quit.")
while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() == "exit":
        break

    ai_response = generate_text(length=700, temperature=0.5, start=user_input)
    print("\nAI:", ai_response[seq_length:])  # Remove seed from output
