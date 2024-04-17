#pip install google-api-python-client
#FOR ADDING LOGOS
#currently not working-- api limit reached = will check tommarow

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
import base64
from googleapiclient.discovery import build  # Add this line

# Set up Spotify credentials
client_id = "609dac35b10a47ee87d9ce8ad8f62b8a"
client_secret = "ae5ab9ca068f4feaa2d40a86d6e86069"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Set up Google API credentials
YOUTUBE_API_KEY = "AIzaSyC7vUv6GFbU_-8fHe1llSAz-moq3gDykgE"
youtube = build("youtube", "v3", developerKey="AIzaSyC7vUv6GFbU_-8fHe1llSAz-moq3gDykgE")

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

def get_recommendations(track_name):
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

            return recommendations, track_url
    else:
        st.write(f"No track found for '{track_name}'")
        return {}, None

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

def load_lottie_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Lottie file: {e}")
        return None

# Load the Lottie animation from the local file
lottie_file_path = r"C:\Users\SIDDHANT GODWANI\Desktop\Streamlit Music R\Loading_Animation.json"
lottie_animation = load_lottie_file(lottie_file_path)

st.title("Music Recommendation System")
track_name = st.text_input("Enter a song name:")

if track_name:
    if lottie_animation is not None:
        with st.spinner(st_lottie(lottie_animation, height=200, key="loading")):
            recommendations, track_url = get_recommendations(track_name)
    else:
        recommendations, track_url = get_recommendations(track_name)

    st.write("Recommended songs:")
    for language, recs in recommendations.items():
        st.write(f"Language: {language}")
        for track, preview_url, spotify_url, youtube_url, image_url in recs:
            st.image(image_url, width=100)
            spotify_logo = base64.b64decode(b'PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDBDNS4zNyAwIDAgNS4zNyAwIDEyQzAgMTguNjMgNS4zNyAyNCAxMiAyNEMxOC42MyAyNCAgMjQgMTguNjMgMjQgMTJDMjQgNS4zNyAxOC42MyAwIDEyIDBaTTEwLjQwNyAxNy4xOThDOS4zMzUgMTcuNDkyIDcuNzM5IDE3LjM2MiA2LjU4NCAxNi41NDdDNS40MjkgMTUuNzMzIDUuMDI1IDE0LjA2NSA2LjA5NyAxMi45NTNDNy4xNjkgMTEuODQgOC43NjUgMTEuNzEyIDEwLjAyIDEyLjQwOUMxMC43ODQgMTIuODY3IDExLjI3NCAxMy41MTkgMTEuNTAzIDE0LjI3OEMxMS43MzIgMTUuMDM2IDExLjcyMiAxNS44NDQgMTEuNDA3IDE2LjU2NEMxMS4wOTMgMTcuMjgzIDEwLjU4IDE3LjgxMiAxMC40MDcgMTcuMTk4Wk0xNS40NSAxNi40NzRDMTQuNDk5IDE2LjkwNyAxMy40MjEgMTcuMDg2IDEyLjM0NyAxNi45OTRDOC4yMjUgMTYuNjMyIDYuNTc5IDE0LjAxNiA3LjgzOCAxMC42ODhDOC45MzkgNy44NzkgMTEuMjE3IDYuNjk1IDE0LjE5IDcuMDIzQzE3LjkzMSA3LjQxNiAxOS41ODEgMTEuNDk4IDE4LjUxOSAxNC4zNTZDMTcuNjYyIDE2Ljc4MSAxNi40IDE2LjA0IDE1LjQ1IDE2LjQ3NFoiIGZpbGw9IiMxREIwNTQiLz4KPC9zdmc+Cg==')
            youtube_logo = base64.b64decode(b'PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIzLjQ5NTYgNi4zOTU3QzIzLjE4NjUgNS4wODY4IDIyLjQzODEgNC4zMzg0IDIxLjEyOTIgNC4wMjkzQzE5LjEyIDMuNSAxMiAzLjUgMTIgMy41QzEyIDMuNSA0Ljg4IDMuNSAyLjg3MDggNC4wMjkzQzEuNTYxOSA0LjMzODQgMC44MTM1IDUuMDg2OCAwLjUwNDQgNi4zOTU3QzAgOC40MjUgMCAxMi4wMDEgMCAxMi4wMDFDMCAxMi4wMDEgMCAxNS41NzYgMC41MDQ0IDE3LjYwNjRDMC44MTM1IDE4LjkxNTMgMS41NjE5IDE5LjY2MzcgMi44NzA4IDE5Ljk3MjhDNC44OCAyMC41IDEyIDIwLjUgMTIgMjAuNUMxMiAyMC41IDE5LjEyIDIwLjUgMjEuMTI5MiAxOS45NzI4QzIyLjQzODEgMTkuNjYzNyAyMy4xODY1IDE4LjkxNTMgMjMuNDk1NiAxNy42MDY0QzI0IDE1LjU3NiAyNCAxMi4wMDEgMjQgMTIuMDAxQzI0IDEyLjAwMSAyNCAgOC40MjUgMjMuNDk1NiA2LjM5NTdaTTE1LjI3MTMgMTIuMDAxTDkuNzc2MiAxNS40ODY0VjguNTE1N0wxNS4yNzEzIDEyLjAwMVoiIGZpbGw9IiNGRjAwMDAiLz4KPC9zdmc+Cg==')
            st.write(f"[<img src='data:image/svg+xml;base64,{spotify_logo.decode()}' width='20' height='20'> {track['name']}]({spotify_url})", unsafe_allow_html=True)
            if preview_url:
                audio_player = st.audio(preview_url)
            else:
                st.write("No preview available.")
            if youtube_url:
                st.write(f"[<img src='data:image/svg+xml;base64,{youtube_logo.decode()}' width='20' height='20'> Listen on YouTube]({youtube_url})", unsafe_allow_html=True)
            else:
                st.write("No YouTube link available.")