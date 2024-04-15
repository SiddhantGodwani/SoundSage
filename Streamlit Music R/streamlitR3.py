# unspecified audio features used (not very efficient)

import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import numpy as np
import tempfile
import requests
import os

# Set up Spotify credentials
client_id = "609dac35b10a47ee87d9ce8ad8f62b8a"
client_secret = "ae5ab9ca068f4feaa2d40a86d6e86069"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define the audio features to extract
audio_features = ['chroma_stft', 'spectral_centroid', 'spectral_bandwidth']

def get_audio_features(preview_url):
    # Create a temporary directory with write permissions
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set the TEMP environment variable to the temporary directory
        old_temp_dir = os.environ.get('TEMP', None)
        os.environ['TEMP'] = temp_dir

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
            response = requests.get(preview_url)
            temp_file.write(response.content)
            temp_file.close()

            y, sr = librosa.load(temp_file.name)
            features = []
            for feature in audio_features:
                feature_value = getattr(librosa.feature, feature)(y=y, sr=sr)
                feature_value = np.reshape(feature_value, (-1, 1))  # Reshape to 2D array
                features.append(feature_value)
            return np.hstack(features)
        finally:
            # Reset the TEMP environment variable
            if old_temp_dir is None:
                del os.environ['TEMP']
            else:
                os.environ['TEMP'] = old_temp_dir

def get_recommendations(track_name):
    # Get track URI and preview URL
    results = sp.search(q=track_name, type='track')
    track_uri = results['tracks']['items'][0]['uri']
    preview_url = results['tracks']['items'][0]['preview_url']

    # Get track audio features
    track_audio_features = get_audio_features(preview_url)

    # Get recommended tracks based on audio features
    recommendations = []
    for seed_track in [track_uri]:
        recs = sp.recommendations(seed_tracks=[seed_track])['tracks']
        for track in recs:
            preview_url = track['preview_url']
            if preview_url:
                audio_features = get_audio_features(preview_url)
                similarity = np.linalg.norm(track_audio_features - audio_features)
                recommendations.append((track, similarity))

    recommendations.sort(key=lambda x: x[1])
    return recommendations[:10]

st.title("Music Recommendation System")
track_name = st.text_input("Enter a song name:")

if track_name:
    recommendations = get_recommendations(track_name)
    st.write("Recommended songs:")
    for track, _ in recommendations:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(track['name'])
        with col2:
            if track['preview_url']:
                st.audio(track['preview_url'])
            else:
                st.write("No preview available")
        with col3:
            st.write(f"[Visit on Spotify]({track['external_urls']['spotify']})")
        st.image(track['album']['images'][0]['url'])