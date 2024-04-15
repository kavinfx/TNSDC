import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop

# Sample dataset of game quest descriptions
quests = [
    "Retrieve the ancient artifact from the Forbidden Temple.",
    "Defend the kingdom from invading monsters.",
    "Explore the mysterious caves and uncover hidden treasures.",
    # Add more quest descriptions as needed
]

# Tokenize the text and prepare sequences for training
tokenizer = Tokenizer()
tokenizer.fit_on_texts(quests)
sequences = tokenizer.texts_to_sequences(quests)
vocab_size = len(tokenizer.word_index) + 1

# Generate input-output pairs for training
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

# Define the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)

# Function to generate quest descriptions
def generate_quest(model, tokenizer, seed_text, max_length):
    result = seed_text
    for _ in range(max_length):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length-1, padding='pre')
        y_pred = np.argmax(model.predict(encoded), axis=-1)
        next_word = tokenizer.index_word[y_pred[0]]
        seed_text += ' ' + next_word
        result += ' ' + next_word
        if next_word == 'end':  # Assuming 'end' is a special token denoting end of quest description
            break
    return result

# Example usage
generated_quest = generate_quest(model, tokenizer, "Begin a journey to", max_length=20)
print("Generated Quest:", generated_quest)
