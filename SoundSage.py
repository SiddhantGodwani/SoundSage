#12.2 but 12.3.1 is also working kinda

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
import urllib.parse
import pickle
import logging
import plotly.graph_objs as go

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

YOUTUBE_CACHE_FILE = 'youtube_cache.pkl'
RECOMMENDATIONS_CACHE_FILE = 'recommendations_cache.csv'

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

def get_cached_youtube_result(query):
    try:
        with open(YOUTUBE_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}

    if query in cache and datetime.now() - cache[query]['timestamp'] < timedelta(days=7):
        return cache[query]['url']
    return None

def cache_youtube_result(query, url):
    try:
        with open(YOUTUBE_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}

    cache[query] = {'url': url, 'timestamp': datetime.now()}

    with open(YOUTUBE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def search_youtube(query):
    cached_result = get_cached_youtube_result(query)
    if cached_result:
        return cached_result

    try:
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=1
        ).execute()

        if search_response["items"]:
            result_url = f"https://www.youtube.com/watch?v={search_response['items'][0]['id']['videoId']}"
        else:
            result_url = None
    except Exception as e:
        logging.error(f"Error searching YouTube: {e}")
        encoded_query = urllib.parse.quote(query)
        result_url = f"https://www.youtube.com/results?search_query={encoded_query}"

    if result_url:
        cache_youtube_result(query, result_url)
    return result_url

def read_recommendations_cache():
    try:
        return pd.read_csv(RECOMMENDATIONS_CACHE_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=['seed_track', 'recommendations', 'timestamp'])

def write_recommendations_cache(seed_track, recommendations):
    cache_df = read_recommendations_cache()
    cache_df = cache_df[cache_df['seed_track'] != seed_track]  # Remove old entry if exists
    new_row = pd.DataFrame({
        'seed_track': [seed_track],
        'recommendations': [json.dumps(recommendations)],
        'timestamp': [datetime.now().isoformat()]
    })
    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    cache_df.to_csv(RECOMMENDATIONS_CACHE_FILE, index=False)

def get_cached_recommendations(seed_track, max_age_days=7):
    cache_df = read_recommendations_cache()
    cached = cache_df[cache_df['seed_track'] == seed_track]
    if not cached.empty:
        timestamp = datetime.fromisoformat(cached.iloc[0]['timestamp'])
        if datetime.now() - timestamp < timedelta(days=max_age_days):
            return json.loads(cached.iloc[0]['recommendations'])
    return None

def get_recommendations(track_name):
    cached_recommendations = get_cached_recommendations(track_name)
    if cached_recommendations:
        return cached_recommendations, None

    results = sp.search(q=f'track:"{track_name}"', type='track')
    if results['tracks']['items']:
        track_uri = results['tracks']['items'][0]['uri']
        track_id = results['tracks']['items'][0]['id']
        track_preview_url = results['tracks']['items'][0]['preview_url']
        track_url = results['tracks']['items'][0]['external_urls']['spotify']
        
        if results['tracks']['items'][0]['album']['available_markets']:
            track_language = results['tracks']['items'][0]['album']['available_markets'][0]
        else:
            track_language = 'Unknown'  # or any default value you prefer
                
        track_image_url = results['tracks']['items'][0]['album']['images'][0]['url']

        track_audio_features = get_audio_features(track_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            track_preview_path = download_preview(track_preview_url, tmp_dir)
            track_librosa_features = get_librosa_features(track_preview_path) if track_preview_path else None

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

                    # Add this check
                    if track['album']['available_markets']:
                        track_language = track['album']['available_markets'][0]
                    else:
                        track_language = 'Unknown'  # or any default value you prefer

                    track_url = track['external_urls']['spotify']
                    track_name = track['name']
                    track_image_url = track['album']['images'][0]['url']
                    youtube_search_query = f"{track_name} {track['artists'][0]['name']}"
                    youtube_video = search_youtube(youtube_search_query)
                    recommendations[track_language].append((track, similarity, preview_url, track_url, youtube_video, track_image_url))


            for language, recs in recommendations.items():
                recs.sort(key=lambda x: x[1])
                recommendations[language] = [(rec[0], rec[2], rec[3], rec[4], rec[5]) for rec in recs]

            user_entered_song = (results['tracks']['items'][0], track_preview_url, track_url, search_youtube(f"{results['tracks']['items'][0]['name']} {results['tracks']['items'][0]['artists'][0]['name']}"), track_image_url)
            recommendations[track_language].insert(0, user_entered_song)

            write_recommendations_cache(track_name, recommendations)

            return recommendations, track_url
    else:
        logging.error(f"No track found for '{track_name}'")
        return {}, None

def display_recommendations(recommendations):
    for language, recs in recommendations.items():
        st.write(f"Language: {language}")
        for i, (track, preview_url, spotify_url, youtube_url, image_url) in enumerate(recs):
            with st.expander(f"{track['name']} - {track['artists'][0]['name']}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(image_url, width=100)
                with col2:
                    st.write(f"[Open in Spotify]({spotify_url})")
                    if preview_url:
                        st.audio(preview_url)
                    else:
                        st.write("No preview available.")
                    if youtube_url:
                        if "youtube.com/watch" in youtube_url:
                            st.write(f"[Listen on YouTube]({youtube_url})")
                        else:
                            st.write(f"[Search on YouTube]({youtube_url})")
                    else:
                        st.write("No YouTube link available.")
                
                rating = st.select_slider(f"Rate '{track['name']}'", options=[1, 2, 3, 4, 5], value=3, key=f"rating_{track['id']}_{i}")
                
                def on_click_callback():
                    threading.Thread(target=store_feedback, args=(track['id'], rating)).start()
                    st.success("Thank you for your feedback!", icon="✅")

                st.button("Submit Rating", key=f"submit_{track['id']}_{i}", on_click=on_click_callback)

    # Visualization
    st.subheader("Audio Features Visualization")
    fig = create_audio_feature_chart(recommendations)
    st.plotly_chart(fig)


def create_audio_feature_chart(recommendations):
    features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
    
    data = []
    for language, recs in recommendations.items():
        for track, _, _, _, _ in recs:
            audio_features = sp.audio_features(track['id'])[0]
            values = [audio_features[feature] for feature in features]
            data.append(go.Scatterpolar(
                r=values,
                theta=features,
                fill='toself',
                name=f"{track['name']} - {track['artists'][0]['name']}"
            ))
    
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig


def store_feedback(track_id, rating):
    feedback_file = 'user_feedback.csv'
    try:
        df = pd.read_csv(feedback_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['track_id', 'rating'])
    
    new_feedback = pd.DataFrame({'track_id': [track_id], 'rating': [rating]})
    df = pd.concat([df, new_feedback], ignore_index=True)
    df.to_csv(feedback_file, index=False)

def clear_old_cache_entries(max_age_days=30):
    cache_df = read_recommendations_cache()
    cache_df['timestamp'] = pd.to_datetime(cache_df['timestamp'])
    cache_df = cache_df[cache_df['timestamp'] > (datetime.now() - timedelta(days=max_age_days))]
    cache_df.to_csv(RECOMMENDATIONS_CACHE_FILE, index=False)

    try:
        with open(YOUTUBE_CACHE_FILE, 'rb') as f:
            youtube_cache = pickle.load(f)
        youtube_cache = {k: v for k, v in youtube_cache.items() if datetime.now() - v['timestamp'] < timedelta(days=max_age_days)}
        with open(YOUTUBE_CACHE_FILE, 'wb') as f:
            pickle.dump(youtube_cache, f)
    except FileNotFoundError:
        pass

def load_lottie_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading Lottie file: {e}")
        return None

# Main Streamlit app
st.title("Music Recommendation System")

# Sidebar for additional options
st.sidebar.header("Options")
max_age_days = st.sidebar.slider("Max age of cached recommendations (days)", 1, 30, 7)

if st.sidebar.button("Clear Old Cache Entries", key="clear_cache"):
    clear_old_cache_entries(max_age_days)
    st.sidebar.success("Old cache entries cleared!")

# Main input
track_name = st.text_input("Enter a song name:", key="song_input")

if track_name:
    lottie_file_path = r"C:\Users\SIDDHANT GODWANI\Desktop\Streamlit Music R\Loading_Animation.json"
    lottie_animation = load_lottie_file(lottie_file_path)

    cached_recommendations = get_cached_recommendations(track_name, max_age_days)
    
    if cached_recommendations:
        st.success("Showing cached recommendations")
        display_recommendations(cached_recommendations)
        
        refresh_button = st.button("Refresh Recommendations", key="refresh_recommendations")
        if refresh_button:
            st.warning("Updating recommendations...")
            recommendations, _ = get_recommendations(track_name)
            st.success("Recommendations updated!")
            display_recommendations(recommendations)
    else:
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
feedback = st.text_area("Please provide any feedback on the recommendations:", key="feedback_input")
if st.button("Submit Feedback", key="submit_feedback"):
    # Here you would typically send this feedback to a database or file
    st.success("Thank you for your feedback!")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.info("""
    This app uses Spotify and YouTube APIs to provide music recommendations.
    It also caches results to improve performance on repeat searches.
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Siddhant Godwani")
