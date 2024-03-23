import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Sample text data
text = "Your text data here..."

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])
vocab_size = len(tokenizer.word_index) + 1

# Generate input-output pairs
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]

# Build LSTM model
model = Sequential([
    Embedding(vocab_size, 50, input_length=X.shape[1]),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)

# Generate text
seed_text = "Your seed text here..."
for _ in range(10):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen=X.shape[1], truncating='pre')
    y_pred = model.predict_classes(encoded, verbose=0)
    out_word = ''
    for word, index in tokenizer.word_index.items():
        if index == y_pred:
            out_word = word
            break
    seed_text += ' ' + out_word
print(seed_text)
