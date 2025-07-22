# 🧠 Next Word Predictor with Streamlit

This is a simple web app that uses a trained LSTM model to predict the **next few words** in a sentence using **deep learning**. It’s built with **TensorFlow**, **Keras**, and **Streamlit**, and supports **live prediction** with a modern UI.

---

## 🚀 Features

- 🔮 Predict up to 5 next words using a trained model
- 🎯 Top 3 predicted completions (beam search style)
- 🖌️ Modern and responsive UI
- 🔁 Live prediction as you type
- ⚡ Fast loading with model caching

---

## 🖥️ Live Demo

You can try the app here: [streamlit.app](https://nwp123.streamlit.app/)

---

## 📂 Project Structure

```
├── app.py                   # Streamlit app
├── NextWordPredictor.ipynb  # Training Model
├── nwp_model.keras          # Trained Keras model
├── tokenizer.pkl            # Tokenizer used for training
├── requirements.txt         # Python dependencies
└── README.md                # (This file)
```

---

## 🛠️ How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/ArmanSinghOP/NWP.git
cd NWP
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this repo to GitHub (public repo required for free tier)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **"New App"** > Connect to your GitHub repo
4. Set `app.py` as the main file and deploy!

> Make sure your model (`.keras`) and `tokenizer.pkl` are both under 100 MB each for free hosting.

---

## 📦 requirements.txt

```
streamlit
tensorflow==2.15.0
numpy
```

---

## 🧠 Model Info

The model is a trained LSTM built on a general English sentence corpus for word-level prediction. Make sure the tokenizer used here matches the one used during training.

---
