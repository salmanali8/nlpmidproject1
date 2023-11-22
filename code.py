import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

import streamlit as st

# Load the CSV file
file_path = '/content/drive/My Drive/dataset2/bbc-text.csv'
df = pd.read_csv(file_path)

# Assuming your CSV has two columns: 'text' and 'category'
X = df['text']
y = df['category']

# Create a text classification pipeline using TF-IDF vectorization and Linear SVM
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Train the model
model.fit(X, y)

# Streamlit app starts here
st.title('Text Category Prediction')
st.write('Enter a sentence to predict its category:')

# Text input for prediction
user_input = st.text_input('Enter a sentence:', '')

# Make predictions on the user input
if user_input:
    prediction = model.predict([user_input])
    st.write(f'Predicted Category: {prediction[0]}')
