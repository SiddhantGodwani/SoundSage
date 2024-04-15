# audio features used but without spotify

import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np

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

def get_recommendations(track_name):
    # Get track URI and ID
    results = sp.search(q=track_name, type='track')
    track_uri = results['tracks']['items'][0]['uri']
    track_id = results['tracks']['items'][0]['id']

    # Get track audio features
    track_audio_features = get_audio_features(track_id)

    # Get recommended tracks based on audio features
    recommendations = []
    for seed_track in [track_uri]:
        recs = sp.recommendations(seed_tracks=[seed_track])['tracks']
        for track in recs:
            track_id = track['id']
            audio_features = get_audio_features(track_id)
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