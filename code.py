import pandas as pd
import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

# Function to download file from Google Drive using file ID
def download_file(file_id):
    downloaded = drive.CreateFile({'id': file_id})
    downloaded.GetContentFile('data.csv')  # Save file locally as 'data.csv'
    return pd.read_csv('data.csv')

# Streamlit app starts here
st.title('Text Category Prediction')
st.write('Google Drive Dataset Access')

# File ID from the provided link
file_id = 'https://drive.google.com/file/d/1cTBJJucUKfaCtr2Jo-oJK9nLILSRht_N/view?usp=drive_link'  # Replace with the file ID from your link

try:
    # Load data from Google Drive using the file ID
    df = download_file(file_id)

    # Assuming your CSV has two columns: 'text' and 'category'
    X = df['text']
    y = df['category']

    # Rest of your code for model training and prediction remains the same...
