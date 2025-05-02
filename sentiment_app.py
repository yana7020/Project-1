import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load a small example dataset
data = {
    'text': [
        'I love this product!',
        'This is the worst experience.',
        'Very happy with the service.',
        'Totally disappointed.',
        'Not bad, just average.',
    ],
    'target': ['positive', 'negative', 'positive', 'negative', 'neutral']
}
df = pd.DataFrame(data)

# Preprocess
X = df['text']
y = df['target']
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Streamlit UI
st.title("Sentiment Analyzer")
st.write("Enter any sentence to predict its sentiment:")

text_input = st.text_input("Your message here:")
if st.button("Analyze"):
    if text_input:
        input_vec = vectorizer.transform([text_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Sentiment: **{prediction.upper()}**")
    else:
        st.warning("Please enter some text.")
