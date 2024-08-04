# feedback loop adaded but error messages shown 


import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import librosa
import requests
import os
import tempfile
from collections import defaultdict
from streamlit_lottie import st_lottie
import json
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime, timedelta
import threading

# Set up Spotify credentials
client_id = "609dac35b10a47ee87d9ce8ad8f62b8a"
client_secret = "ae5ab9ca068f4feaa2d40a86d6e86069"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Set up Google API credentials
YOUTUBE_API_KEY = "AIzaSyC7vUv6GFbU_-8fHe1llSAz-moq3gDykgE"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Define the audio features to extract
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def get_audio_features(track_id):
    track_features = sp.audio_features(track_id)[0]
    features = [track_features[feature] for feature in audio_features]
    return np.array(features)

def get_librosa_features(track_preview_path):
    y, sr = librosa.load(track_preview_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    return mfcc, chroma, spectral_centroid, spectral_rolloff, tempogram, zcr, rms

def download_preview(track_preview_url, tmp_dir):
    if track_preview_url:
        response = requests.get(track_preview_url)
        if response.status_code == 200:
            tmp_file = os.path.join(tmp_dir, 'preview.mp3')
            with open(tmp_file, 'wb') as f:
                f.write(response.content)
            return tmp_file
    return None

def search_youtube(query):
    try:
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=1
        ).execute()

        if search_response["items"]:
            return f"https://www.youtube.com/watch?v={search_response['items'][0]['id']['videoId']}"
        else:
            return None
    except Exception as e:
        st.error(f"Error searching YouTube: {e}")
        return None

def read_recommendations_cache(file_name='recommendations_cache.csv'):
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        return pd.DataFrame(columns=['seed_track', 'recommendations', 'timestamp'])

def write_recommendations_cache(seed_track, recommendations, file_name='recommendations_cache.csv'):
    cache_df = read_recommendations_cache(file_name)
    new_row = pd.DataFrame({
        'seed_track': [seed_track],
        'recommendations': [json.dumps(recommendations)],
        'timestamp': [datetime.now().isoformat()]
    })
    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    cache_df.to_csv(file_name, index=False)

def get_cached_recommendations(seed_track, max_age_days=7):
    cache_df = read_recommendations_cache()
    cached = cache_df[cache_df['seed_track'] == seed_track]
    if not cached.empty:
        timestamp = datetime.fromisoformat(cached.iloc[0]['timestamp'])
        if datetime.now() - timestamp < timedelta(days=max_age_days):
            return json.loads(cached.iloc[0]['recommendations'])
    return None

def get_recommendations(track_name):
    # Check cache first
    cached_recommendations = get_cached_recommendations(track_name)
    if cached_recommendations:
        return cached_recommendations, None  # Return cached recommendations immediately

    # Get track URI and ID
    results = sp.search(q=f'track:"{track_name}"', type='track')
    if results['tracks']['items']:
        track_uri = results['tracks']['items'][0]['uri']
        track_id = results['tracks']['items'][0]['id']
        track_preview_url = results['tracks']['items'][0]['preview_url']
        track_url = results['tracks']['items'][0]['external_urls']['spotify']
        track_language = results['tracks']['items'][0]['album']['available_markets'][0]
        track_image_url = results['tracks']['items'][0]['album']['images'][0]['url']

        # Get track audio features
        track_audio_features = get_audio_features(track_id)

        # Create a temporary directory for preview downloads
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the seed track preview
            track_preview_path = download_preview(track_preview_url, tmp_dir)
            track_librosa_features = get_librosa_features(track_preview_path) if track_preview_path else None

            # Get recommended tracks based on audio features
            recommendations = defaultdict(list)
            for seed_track in [track_uri]:
                recs = sp.recommendations(seed_tracks=[seed_track])['tracks']
                for track in recs:
                    track_id = track['id']
                    audio_features = get_audio_features(track_id)
                    preview_url = track['preview_url']
                    preview_path = download_preview(preview_url, tmp_dir)
                    if preview_path:
                        librosa_features = get_librosa_features(preview_path)
                        similarity = np.linalg.norm(track_audio_features - audio_features) + sum(np.linalg.norm(feature1 - feature2) for feature1, feature2 in zip(track_librosa_features, librosa_features))
                    else:
                        similarity = np.linalg.norm(track_audio_features - audio_features)
                    track_language = track['album']['available_markets'][0]
                    track_url = track['external_urls']['spotify']
                    track_name = track['name']
                    track_image_url = track['album']['images'][0]['url']
                    youtube_search_query = f"{track_name} {track['artists'][0]['name']}"
                    youtube_video = search_youtube(youtube_search_query)
                    recommendations[track_language].append((track, similarity, preview_url, track_url, youtube_video, track_image_url))

            # Sort recommendations by language and similarity
            for language, recs in recommendations.items():
                recs.sort(key=lambda x: x[1])
                recommendations[language] = [(rec[0], rec[2], rec[3], rec[4], rec[5]) for rec in recs]

            # Display the user-entered song first
            user_entered_song = (results['tracks']['items'][0], track_preview_url, track_url, search_youtube(f"{results['tracks']['items'][0]['name']} {results['tracks']['items'][0]['artists'][0]['name']}"), track_image_url)
            recommendations[track_language].insert(0, user_entered_song)

            # Store the recommendations in the cache
            write_recommendations_cache(track_name, recommendations)

            return recommendations, track_url
    else:
        st.write(f"No track found for '{track_name}'")
        return {}, None

def display_recommendations(recommendations):
    for language, recs in recommendations.items():
        st.write(f"Language: {language}")
        for track, preview_url, spotify_url, youtube_url, image_url in recs:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(image_url, width=100)
            with col2:
                st.write(f"[{track['name']}]({spotify_url})")
                if preview_url:
                    st.audio(preview_url)
                else:
                    st.write("No preview available.")
                if youtube_url:
                    st.write(f"[Listen on YouTube]({youtube_url})")
                else:
                    st.write("No YouTube link available.")
                
                # Add rating system
                rating = st.select_slider(f"Rate '{track['name']}'", options=[1, 2, 3, 4, 5], value=3, key=f"rating_{track['id']}")
                if st.button("Submit Rating", key=f"submit_{track['id']}"):
                    store_feedback(track['id'], rating)
                    st.success("Thank you for your feedback!")

def store_feedback(track_id, rating):
    feedback_file = 'user_feedback.csv'
    try:
        df = pd.read_csv(feedback_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['track_id', 'rating'])
    
    new_feedback = pd.DataFrame({'track_id': [track_id], 'rating': [rating]})
    df = pd.concat([df, new_feedback], ignore_index=True)
    df.to_csv(feedback_file, index=False)

def background_update(track_name):
    fresh_recommendations, _ = get_recommendations(track_name)
    # You might want to implement a way to notify the user that fresh recommendations are available

def load_lottie_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Lottie file: {e}")
        return None

# Main Streamlit app
st.title("Music Recommendation System")
track_name = st.text_input("Enter a song name:")

if track_name:
    # Load the Lottie animation
    lottie_file_path = r"C:\Users\SIDDHANT GODWANI\Desktop\Streamlit Music R\Loading_Animation.json"
    lottie_animation = load_lottie_file(lottie_file_path)

    # Display cached recommendations immediately if available
    cached_recommendations, _ = get_recommendations(track_name)
    if cached_recommendations:
        st.write("Cached Recommendations:")
        display_recommendations(cached_recommendations)
    
    # Start background processing for fresh recommendations
    if lottie_animation is not None:
        with st.spinner(st_lottie(lottie_animation, height=200, key="loading")):
            fresh_recommendations, track_url = get_recommendations(track_name)
    else:
        with st.spinner("Updating recommendations..."):
            fresh_recommendations, track_url = get_recommendations(track_name)
    
    # Display fresh recommendations if they're different from cached ones
    if fresh_recommendations != cached_recommendations:
        st.write("Updated Recommendations:")
        display_recommendations(fresh_recommendations)

    # Start background processing
    threading.Thread(target=background_update, args=(track_name,)).start()
    
    # ... (previous code remains the same)

# Main Streamlit app
st.title("Music Recommendation System")

# Sidebar for additional options
st.sidebar.header("Options")
max_age_days = st.sidebar.slider("Max age of cached recommendations (days)", 1, 30, 7)

# Main input
track_name = st.text_input("Enter a song name:")

if track_name:
    # Load the Lottie animation
    lottie_file_path = r"C:\Users\SIDDHANT GODWANI\Desktop\Streamlit Music R\Loading_Animation.json"
    lottie_animation = load_lottie_file(lottie_file_path)

    # Check for cached recommendations
    cached_recommendations = get_cached_recommendations(track_name, max_age_days)
    
    if cached_recommendations:
        st.success("Showing cached recommendations")
        display_recommendations(cached_recommendations)
        
        # Start background processing for fresh recommendations
        refresh_button = st.button("Refresh Recommendations")
        if refresh_button:
            st.warning("Updating recommendations...")
            fresh_recommendations, _ = get_recommendations(track_name)
            st.success("Recommendations updated!")
            display_recommendations(fresh_recommendations)
    else:
        # If no cached recommendations, process new ones
        with st.spinner("Getting recommendations..."):
            if lottie_animation:
                st_lottie(lottie_animation, height=200, key="loading")
            recommendations, _ = get_recommendations(track_name)
        
        if recommendations:
            st.success("Here are your recommendations:")
            display_recommendations(recommendations)
        else:
            st.error("Unable to find recommendations for this track. Please try another.")

# Feedback section
st.header("Help us improve!")
feedback = st.text_area("Please provide any feedback on the recommendations:")
if st.button("Submit Feedback"):
    # Here you would typically send this feedback to a database or file
    # For now, we'll just acknowledge it
    st.success("Thank you for your feedback!")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.info("""
    This app uses Spotify and YouTube APIs to provide music recommendations.
    It also caches results to improve performance on repeat searches.
    """)

# You might want to add a footer
st.markdown("---")
st.markdown("Created with ❤️ by Your SID")