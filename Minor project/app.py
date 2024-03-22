import os
from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import spotipy
import tempfile
import requests
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation

app = Flask(__name__, template_folder='templates')

# Set up Spotify API credentials
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get audio data and extract features for the liked songs
def get_liked_songs_features(song_names, user_preferences):
    liked_songs_features = []
    song_ids = []
    for song_name in song_names:
        try:
            results = sp.search(q=song_name, type='track', limit=1)
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                song_id = track['id']
                song_ids.append(song_id)
                preview_url = track['preview_url']
                spotify_features = sp.audio_features(tracks=[song_id])[0]
                genre = user_preferences.get(song_id, 'unknown')
                if preview_url:
                    try:
                        response = requests.get(preview_url)
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_file.write(response.content)
                            temp_file_path = temp_file.name
                        y, sr = librosa.load(temp_file_path, offset=10, duration=30)
                        os.remove(temp_file_path)
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                        feature_vector = [mfccs.mean(axis=1), spectral_contrast.mean(axis=1), chroma_stft.mean(axis=1)]
                        feature_vector.extend([spotify_features[f] for f in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
                        feature_vector.append(genre)
                        liked_songs_features.append(feature_vector)
                    except Exception as e:
                        print(f"Error loading preview for '{track['name']}' by {track['artists'][0]['name']}: {e}")
                else:
                    print(f"No preview available for '{track['name']}' by {track['artists'][0]['name']}")
            else:
                print(f"No results found for '{song_name}'")
        except Exception as e:
            print(f"Error processing '{song_name}': {e}")
    return liked_songs_features, song_ids

# Get recommendations and filter them based on audio features
def get_filtered_recommendations(liked_songs_features, song_ids):
    recommendations = []
    recommended_songs_features = []
    recommended_songs_ids = []
    for song_id in song_ids:
        recs = sp.recommendations(seed_tracks=[song_id], limit=10, market=None)
        recommendations.extend(recs['tracks'])

    for track in recommendations:
        preview_url = track['preview_url']
        spotify_features = sp.audio_features(tracks=[track['id']])[0]
        if preview_url:
            y, sr = librosa.load(preview_url, offset=10, duration=30)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            feature_vector = [mfccs.mean(axis=1), spectral_contrast.mean(axis=1), chroma_stft.mean(axis=1)]
            feature_vector.extend([spotify_features[f] for f in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
            feature_vector.append('unknown')
            recommended_songs_features.append(feature_vector)
            recommended_songs_ids.append(track['id'])
        else:
            print(f"No preview available for '{track['name']}' by {track['artists'][0]['name']}")

    nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='correlation').fit(liked_songs_features)
    filtered_recommendations = []
    for features in recommended_songs_features:
        distances, indices = nn_model.kneighbors([features])
        if np.min(distances) < 0.5:
            filtered_recommendations.append(recommended_songs_ids[recommended_songs_features.index(features)])

    return filtered_recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song_names = request.json['likedSongs']
    liked_songs_features, song_ids = get_liked_songs_features(song_names, {})
    filtered_recommendations = get_filtered_recommendations(liked_songs_features, song_ids)
    recommended_tracks = []
    for song_id in filtered_recommendations:
        track = sp.track(song_id)
        recommended_tracks.append({
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'preview_url': track['preview_url']
        })
    return jsonify(recommended_tracks)

if __name__ == '__main__':
    app.run(debug=True)