import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

# Load and preprocess text
filepath = "filtered_physics.txt"
text = open(filepath, "rb").read().decode(encoding="utf-8").lower()

with open(filepath, "r", encoding="utf-8") as f:
    text = ""
    for line in f:
        if line.startswith("<PARA>"):
            line = line.replace("<PARA>", "").replace("<END>", "").strip()
            text += line + " § "  # Add stop token after each paragraph

# Create char-to-index mappings
chars = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

# Generate training sequences
seq_length = 60
step_size = 3
sentences = []
next_char = []

for i in range(0, len(text) - seq_length, step_size):
    sentences.append(text[i : i + seq_length])
    next_char.append(text[i + seq_length])

# One-hot encode inputs and targets
x = np.zeros((len(sentences), seq_length, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# Build the model

from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(seq_length, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(len(chars), activation="softmax"))
# ⚠️ Important: Fix optimizer creation
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.004))

# Train the model
model.fit(x, y, batch_size=256, epochs=10)


model.save("phy_three.h5")


import pickle

with open("char_mappings.pkl", "wb") as f:
    pickle.dump((char_to_index, index_to_char), f)


# Sampling function
def sample(preds, temperature=1.6):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature  # add epsilon to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Text generation
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - seq_length - 1)
    sentence = text[start_index : start_index + seq_length]
    generated = sentence
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


# Generate and print text
print(generate_text(300, 0.6))
