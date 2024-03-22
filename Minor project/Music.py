import os
import librosa
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.neighbors import NearestNeighbors
import requests

# Set up Spotify API credentials
client_id = '609dac35b10a47ee87d9ce8ad8f62b8a'
client_secret = 'ae5ab9ca068f4feaa2d40a86d6e86069'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get audio data and extract features for the liked songs
def get_liked_songs_features(liked_songs, user_preferences):
    liked_songs_features = []
    for song_id in liked_songs:
        track = sp.track(song_id)
        preview_url = track['preview_url']
        spotify_features = sp.audio_features(tracks=[song_id])[0]
        genre = user_preferences.get(song_id, 'unknown')
        if preview_url:
            try:
                # Load the downloaded audio file using librosa
                y, sr = librosa.load(preview_url, sr=None, mono=True, offset=10, duration=30)

                # Extract features and append to the list
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                feature_vector = mfccs.mean(axis=1)
                feature_vector = np.concatenate([feature_vector, [spotify_features[f] for f in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]])
                feature_vector = np.concatenate([feature_vector, ['unknown']])
                liked_songs_features.append(feature_vector)
            except Exception as e:
                print(f"Error processing '{track['name']}' by {track['artists'][0]['name']}: {e}")
        else:
            print(f"No preview available for '{track['name']}' by {track['artists'][0]['name']}. Filling in missing values with 0.")
            feature_vector = np.zeros(20 + 11 + 1)
            feature_vector[-1] = 1  # Set genre to 'unknown'
            liked_songs_features.append(feature_vector)
    return liked_songs_features

# Get recommendations and filter them based on audio features
def get_filtered_recommendations(liked_songs, liked_songs_features):
    recommendations = []
    recommended_songs_features = []
    recommended_songs_ids = []
    for song_id in liked_songs:
        recs = sp.recommendations(seed_tracks=[song_id], limit=10, market=None)
        recommendations.extend(recs['tracks'])

    for track in recommendations:
        preview_url = track['preview_url']
        spotify_features = sp.audio_features(tracks=[track['id']])[0]
        if preview_url:
            try:
                # Load the downloaded audio file using librosa
                y, sr = librosa.load(preview_url, sr=None, mono=True, offset=10, duration=30)

                # Extract features and append to the list
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                feature_vector = mfccs.mean(axis=1)
                feature_vector = np.concatenate([feature_vector, [spotify_features[f] for f in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]])
                feature_vector = np.concatenate([feature_vector, ['unknown']])
                recommended_songs_features.append(feature_vector)
                recommended_songs_ids.append(track['id'])
            except Exception as e:
                print(f"Error processing '{track['name']}' by {track['artists'][0]['name']}: {e}")
        else:
            print(f"No preview available for '{track['name']}' by {track['artists'][0]['name']}")

    nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='correlation').fit(liked_songs_features)
    filtered_recommendations = []
    for features in recommended_songs_features:
        distances, indices = nn_model.kneighbors([features])
        if np.min(distances) < 0.5:  # Adjust this threshold as needed
            filtered_recommendations.append(recommended_songs_ids[recommended_songs_features.index(features)])

    # Ensure at least 10 recommendations
    if len(filtered_recommendations) < 10:
        return recommended_songs_ids[:10]
    else:
        return filtered_recommendations

# Update the model based on user feedback
def update_model_with_feedback(liked_songs_features, recommended_songs_features, recommended_songs_ids, user_feedback):
    for song_id, liked in user_feedback.items():
        if liked:
            liked_songs_features.append(recommended_songs_features[recommended_songs_ids.index(song_id)])
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='correlation').fit(liked_songs_features)
    return nn_model

def main():
    liked_songs = []
    user_preferences = {}
    liked_songs_features = []
    nn_model = None

    while True:
        print("\nOptions:")
        print("1. Add a liked song")
        print("2. Get song recommendations")
        print("3. Provide feedback on recommendations")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            song_name = input("Enter the name of a song you like: ")
            results = sp.search(q=song_name, type='track', limit=1)
            if results['tracks']['items']:
                song_id = results['tracks']['items'][0]['id']
                liked_songs.append(song_id)
                genre = input(f"Enter the genre for '{results['tracks']['items'][0]['name']}' (or 'skip'): ")
                if genre.lower() != 'skip':
                    user_preferences[song_id] = genre
            else:
                print(f"No results found for '{song_name}'")

        elif choice == '2':
            if not liked_songs:
                print("Please add some liked songs first.")
            else:
                liked_songs_features = get_liked_songs_features(liked_songs, user_preferences)
                filtered_recommendations = get_filtered_recommendations(liked_songs, liked_songs_features)
                print("\nRecommended songs:")
                for song_id in filtered_recommendations:
                    track = sp.track(song_id)
                    print(f"{track['artists'][0]['name']} - {track['name']}")

        elif choice == '3':
            if not liked_songs:
                print("Please add some liked songs and get recommendations first.")
            else:
                recommended_songs_features = []
                recommended_songs_ids = []
                for song_id in filtered_recommendations:
                    track = sp.track(song_id)
                    preview_url = track['preview_url']
                    spotify_features = sp.audio_features(tracks=[song_id])[0]
                    if preview_url:
                        try:
                            # Load the downloaded audio file using librosa
                            y, sr = librosa.load(preview_url, sr=None, mono=True, offset=10, duration=30)

                            # Extract features and append to the list
                            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                            feature_vector = mfccs.mean(axis=1)
                            feature_vector = np.concatenate([feature_vector, [spotify_features[f] for f in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]])
                            feature_vector = np.concatenate([feature_vector, ['unknown']])
                            recommended_songs_features.append(feature_vector)
                            recommended_songs_ids.append(song_id)
                        except Exception as e:
                            print(f"Error processing '{track['name']}' by {track['artists'][0]['name']}: {e}")
                    else:
                        print(f"No preview available for '{track['name']}' by {track['artists'][0]['name']}")

                user_feedback = {}
                for song_id in filtered_recommendations:
                    track = sp.track(song_id)
                    feedback = input(f"Did you like '{track['name']}' by {track['artists'][0]['name']}? (y/n): ")
                    user_feedback[song_id] = feedback.lower() == 'y'

                nn_model = update_model_with_feedback(liked_songs_features, recommended_songs_features, recommended_songs_ids, user_feedback)
                print("Model updated based on your feedback.")

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
