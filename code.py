import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Streamlit app starts here
st.title('Text Category Prediction')
st.markdown('### Predict the Category of a Sentence')

# Raw URL of the CSV file in your GitHub repository
file_url = 'https://raw.githubusercontent.com/salmanali8/nlpmidproject1/CODE-WITH-SA/bbc-text.csv'  # Replace with your raw CSV file URL

try:
    # Load data from GitHub URL
    df = pd.read_csv(file_url)

    # Assuming your CSV has two columns: 'text' and 'category'
    X = df['text']
    y = df['category']

    # Create a text classification pipeline using TF-IDF vectorization and Linear SVM
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

    # Train the model
    model.fit(X, y)

    # Text input for prediction
    st.sidebar.header('Enter a sentence to predict its category:')
    user_input = st.sidebar.text_input('', '')

    # Make predictions based on user input
    if user_input:
        prediction = model.predict([user_input])
        st.success(f'Predicted Category: {prediction[0]}')

except Exception as e:
    st.write("Error:", e)
