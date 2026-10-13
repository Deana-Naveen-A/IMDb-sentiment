"""Streamlit demo. Run: streamlit run app.py"""
import streamlit as st

from src.predict import (
    bert_available,
    load_bert,
    load_classical,
    predict_bert,
    predict_classical,
)

st.set_page_config(page_title="IMDb Sentiment Classifier", page_icon="🎬")
st.title("🎬 IMDb Review Sentiment Classifier")
st.caption("UE24CS352A Machine Learning mini-project - paste a movie review and get a prediction.")

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "Linear SVM": "linear_svm",
}
options = list(MODELS) + (["DistilBERT"] if bert_available() else [])
choice = st.sidebar.selectbox("Model", options)


@st.cache_resource
def get_classical(name):
    return load_classical(name)


@st.cache_resource
def get_bert():
    return load_bert()


EXAMPLES = {
    "Clearly positive": "An absolute masterpiece. The acting was superb and the story kept me hooked until the very end.",
    "Clearly negative": "Terrible movie. The plot made no sense, the acting was wooden and I wanted my money back.",
    "Negation": "This film is not good at all. I would not recommend it to anyone.",
    "Tricky / sarcastic": "Oh great, another two hours of my life I will never get back. Truly a cinematic 'gem'.",
}
st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label):
        st.session_state["review"] = text

review = st.text_area("Review text", key="review", height=180)

if st.button("Predict", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        if choice == "DistilBERT":
            label, conf = predict_bert(get_bert(), review)
        else:
            label, conf = predict_classical(get_classical(MODELS[choice]), review)
        if label == "Positive":
            st.success(f"Positive 😊 (confidence {conf:.1%})")
        else:
            st.error(f"Negative 😞 (confidence {conf:.1%})")
        st.progress(conf)
        if choice == "Linear SVM":
            st.caption("SVM confidence is a sigmoid of the margin, so it is not a calibrated probability.")
