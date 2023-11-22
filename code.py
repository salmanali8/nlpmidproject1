import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import streamlit as st

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

# Function to get file from Google Drive
def get_file_from_drive(file_id):
    downloaded = drive.CreateFile({'id': file_id})
    downloaded.GetContentFile('bbc-text.csv')  # Save file locally as 'bbc-text.csv'
    return pd.read_csv('bbc-text.csv')

# Streamlit app starts here
st.title('Text Category Prediction')
st.write('Google Drive Dataset Access')

# File ID of your dataset in Google Drive
file_id = 'YOUR_FILE_ID_HERE'  # Replace with the actual file ID from your Google Drive

try:
    # Load data from Google Drive
    df = get_file_from_drive(file_id)

    # Assuming your CSV has two columns: 'text' and 'category'
    X = df['text']
    y = df['category']

    # Create a text classification pipeline using TF-IDF vectorization and Linear SVM
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

    # Train the model
    model.fit(X, y)

    st.write('Data loaded from Google Drive and model trained!')

    # Text input for prediction
    st.write('Enter a sentence to predict its category:')
    user_input = st.text_input('Enter a sentence:', '')

    # Make predictions on the user input
    if user_input:
        prediction = model.predict([user_input])
        st.write(f'Predicted Category: {prediction[0]}')
except Exception as e:
    st.write("Error:", e)

