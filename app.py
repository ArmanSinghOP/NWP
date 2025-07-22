import streamlit as st
import numpy as np
import pickle
import tensorflow.keras as keras

# ---- Page config ----
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

# ---- Custom CSS for modern feel ----
st.markdown("""
    <style>
        .main {
            font-family: 'Segoe UI', sans-serif;
        }
        .prediction {
            font-size: 18px;
            margin-top: 1rem;
        }
        .predicted-text {
            color: #0072C6;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Cache model & tokenizer ----
@st.cache_resource
def load_model():
    return keras.models.load_model("nwp_model.keras", compile=False)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_sequence_len = model.input_shape[1]
index_word = {index: word for word, index in tokenizer.word_index.items()}

# ---- Function to get top N predictions ----
def predict_top_n(seed_text, next_words, top_n=3):
    seed_text = seed_text.strip()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = token_list[-(max_sequence_len - 1):]

    if not token_list:
        return None, "❌ Unable to predict due to unknown words."

    padded = keras.preprocessing.sequence.pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding='pre'
    )
    pred = model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(pred)[-top_n:][::-1]

    completions = []
    for idx in top_indices:
        word = index_word.get(idx, "")
        new_text = seed_text + ' ' + word
        generated = new_text
        for _ in range(next_words - 1):  # already added 1 word
            sub_tokens = tokenizer.texts_to_sequences([generated])[0][-max_sequence_len + 1:]
            sub_padded = keras.preprocessing.sequence.pad_sequences(
                [sub_tokens], maxlen=max_sequence_len - 1, padding='pre'
            )
            next_pred = model.predict(sub_padded, verbose=0)
            next_word_idx = np.argmax(next_pred, axis=-1)[0]
            next_word = index_word.get(next_word_idx, "")
            generated += ' ' + next_word
        completions.append(generated)
    return completions, None

# ---- UI ----
st.markdown("<h1 style='text-align:center; color:#0072C6;'>🧠 Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>LSTM based next word generator</p>", unsafe_allow_html=True)

user_input = st.text_input("Enter your text:", placeholder="e.g. He could have")
num_words = st.number_input("How many words to predict?", min_value=1, max_value=5, value=3, step=1)

# ---- Show predictions ----
if user_input.strip():
    completions, error = predict_top_n(user_input, num_words, top_n=3)

    if error:
        st.warning(error)
    else:
        st.markdown("### 🔮 Top 3 Predictions")
        for i, comp in enumerate(completions, start=1):
            original_len = len(user_input.strip().split())
            words = comp.split()
            original = ' '.join(words[:original_len])
            predicted = ' '.join(words[original_len:])
            st.markdown(
                f"<div class='prediction'>#{i}: {original} <span class='predicted-text'>{predicted}</span></div>",
                unsafe_allow_html=True
            )

# ---- Footer ----
st.markdown("---")
st.caption("🚀 Powered by Streamlit & TensorFlow ")
