import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load model and tokenizer
model = tf.keras.models.load_model("nwp_model.keras", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1]

# Function to generate next words
def predict_next_words(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-(max_sequence_len-1):]  # Truncate if needed
        padded = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(padded, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                seed_text += ' ' + word
                break
    return seed_text

# Streamlit UI
st.title("🧠 Next Word Predictor")
st.markdown("Enter a text prompt, and let the model generate up to 5 next words.")

user_input = st.text_input("Enter your text:")
num_words = st.slider("How many words to predict?", 1, 5, 1)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_next_words(user_input, num_words)
        st.success(f"**Predicted Text:** {result}")
