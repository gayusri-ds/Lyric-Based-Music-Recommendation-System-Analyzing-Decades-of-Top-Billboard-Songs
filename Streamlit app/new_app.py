import streamlit as st
import pandas as pd
import re
import joblib
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors



# Load the data and models
data = pd.read_csv(r'C:/Users/DELL/Desktop/My Projects/Recommendation system/Dataset/all_songs_data.csv')
model = joblib.load(r'C:/Users/DELL/Desktop/My Projects/Recommendation system/saved models/knn_model.pkl')
vectorizer = joblib.load(r'C:/Users/DELL/Desktop/My Projects/Recommendation system/saved models/vectorizer_knn.pkl')

# Normalize the text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

# Function to extract URL from Media column
def extract_url(text):
    media_list = ast.literal_eval(text)
    if media_list and isinstance(media_list, list):
        return media_list[0].get('url', 'unknown')
    return 'unknown'

# Fill missing values and normalize text
data['Album'] = data['Album'].fillna('Unknown Album')
data['Featured Artists'] = data['Featured Artists'].fillna('None')
data['Release Date'] = data['Release Date'].fillna('Unknown')
data['Writers'] = data['Writers'].fillna('Unknown Writers')
data['Year'] = data['Year'].fillna(data['Year'].median())
data = data.dropna(subset=['Album URL', 'Artist', 'Lyrics', 'Media', 'Song URL'])

data['Album'] = data['Album'].apply(normalize_text)
data['Artist'] = data['Artist'].apply(normalize_text)
data['Featured Artists'] = data['Featured Artists'].apply(normalize_text)
data['Lyrics'] = data['Lyrics'].apply(normalize_text)
data['Song Title'] = data['Song Title'].apply(normalize_text)
data['Writers'] = data['Writers'].apply(normalize_text)
data['Media'] = data['Media'].apply(extract_url)

# Feature extraction
data['combined_features'] = data['Song Title'] + ' ' + data['Album'] + ' ' + data['Artist'] + ' ' + \
                             data['Featured Artists'] + ' ' + data['Writers'] + ' ' + data['Lyrics']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
x = vectorizer.fit_transform(data['combined_features'])

# Streamlit app title and description
st.title('🎵 Song Recommendation System 🎶')
st.markdown("""
Welcome to the Song Recommendation System! Enter the name of your favorite song, 
and we'll suggest some tunes you might love. Enjoy discovering new music! 🎧
""")

# Sidebar for user input
st.sidebar.title('Search Your Favorite Song')
st.sidebar.markdown("Type in a song name and click on 'Recommend' to get suggestions.")

# User input for song name
song_name = st.sidebar.text_input('Enter your favorite song name:', '')

# User input for number of recommendations
num_recommendations = st.sidebar.slider('How many recommendations would you like?', 1, 20, 10)

if st.sidebar.button('Recommend'):
    # Normalize the input song name
    normalized_song = normalize_text(song_name)
    
    # Check if the song is in the dataset
    if normalized_song not in data['Song Title'].values:
        st.write(f"⚠️ Sorry, we couldn't find the song '{song_name}' in the dataset. Please try another song.")
    else:
        # Find the index of the song in the dataset
        song_index = data[data['Song Title'] == normalized_song].index[0]
        
        # Transform the song's combined features using the vectorizer
        song_vector = vectorizer.transform([data.iloc[song_index]['combined_features']])
        
        # Find the nearest neighbors
        distances, indices = model.kneighbors(song_vector, n_neighbors=num_recommendations)
        
        # Get the recommendations
        recommendations = data.iloc[indices[0]].copy()
        recommendations['Similarity'] = 1 - distances[0]
        
        # Display recommendations in DataFrame
        st.write(f"### Here are your top {num_recommendations} recommendations:")
        st.write(recommendations[['Song Title', 'Artist', 'Similarity', 'Album URL', 'Song URL']].reset_index(drop=True).to_html(escape=False, render_links=True), unsafe_allow_html=True)
        
        st.balloons()
