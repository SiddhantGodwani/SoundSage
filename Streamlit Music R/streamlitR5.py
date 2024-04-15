# More audio features added
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import librosa
import requests
import os
import tempfile

# Set up Spotify credentials
client_id = "609dac35b10a47ee87d9ce8ad8f62b8a"
client_secret = "ae5ab9ca068f4feaa2d40a86d6e86069"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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
    results = sp.search(q=track_name, type='track')
    track_uri = results['tracks']['items'][0]['uri']
    track_id = results['tracks']['items'][0]['id']
    track_preview_url = results['tracks']['items'][0]['preview_url']

    # Get track audio features
    track_audio_features = get_audio_features(track_id)

    # Create a temporary directory for preview downloads
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download the seed track preview
        track_preview_path = download_preview(track_preview_url, tmp_dir)
        track_librosa_features = get_librosa_features(track_preview_path) if track_preview_path else None

        # Get recommended tracks based on audio features
        recommendations = []
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
                recommendations.append((track, similarity))
        recommendations.sort(key=lambda x: x[1])
        return [rec[0] for rec in recommendations[:10]]

st.title("Music Recommendation System")
track_name = st.text_input("Enter a song name:")
if track_name:
    recommendations = get_recommendations(track_name)
    st.write("Recommended songs:")
    for track in recommendations:
        st.write(track['name'])
        st.image(track['album']['images'][0]['url'])