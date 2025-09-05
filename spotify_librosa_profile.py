import pandas as pd
import librosa
import numpy as np
import os
import spotipy
import time
import yt_dlp
import argparse # Import the argparse library
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

# (The extract_librosa_features and download_audio_for_track functions are unchanged)
def extract_librosa_features(file_path):
    """
    Extracts a comprehensive set of musical features from a local audio file.
    """
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        features = {}
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'librosa_mfcc{i}_mean'] = np.mean(mfccs[i-1])
            features[f'librosa_mfcc{i}_std'] = np.std(mfccs[i-1])
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['librosa_spectral_centroid_mean'] = np.mean(spec_cent)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['librosa_spectral_rolloff_mean'] = np.mean(spec_rolloff)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['librosa_zcr_mean'] = np.mean(zcr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['librosa_chroma_mean'] = np.mean(chroma)
        tempo_values = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        features['librosa_tempo'] = np.mean(tempo_values) if tempo_values.size > 0 else 0
        rms = librosa.feature.rms(y=y)
        features['librosa_rms_energy_mean'] = np.mean(rms)
        return features
    except Exception as e:
        print(f"\nLibrosa/File Error processing {os.path.basename(file_path)}: {e}")
        return None

def download_audio_for_track(track_name, artist_name):
    """
    Searches YouTube for a track and downloads a short audio clip.
    Returns the path to the downloaded file, or None if it fails.
    """
    search_query = f"{artist_name} - {track_name} official audio"
    temp_filename = "temp_audio.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': 'temp_audio',
        'noplaylist': True,
        'default_search': 'ytsearch1',
        'quiet': True,
        'no_warnings': True,
        'postprocessor_args': ['-ss', '00:00:30.00', '-t', '00:00:30.00']
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])
        if os.path.exists(temp_filename):
            return temp_filename
        else:
            return None
    except Exception as e:
        return None

def create_full_audio_profile(all_tracks, output_csv): # Now takes all_tracks as an argument
    """
    Generates a profile using yt-dlp to source audio for Librosa analysis.
    """
    # --- Step 2: Process Each Track using yt-dlp for Librosa Features ---
    all_tracks_data = []
    track_ids = []
    
    for item in tqdm(all_tracks, desc="Processing Songs with yt-dlp"):
        track = item['track']
        if not track or not track.get('id'): continue
        
        track_ids.append(track['id'])
        artist_name = ", ".join([artist['name'] for artist in track['artists']])
        track_name = track['name']
        
        track_data = {'song_name': track_name, 'artist': artist_name, 'spotify_id': track['id'], 'popularity': track['popularity']}
        
        audio_file_path = download_audio_for_track(track_name, artist_name)
        
        if audio_file_path:
            try:
                librosa_features = extract_librosa_features(audio_file_path)
                if librosa_features:
                    track_data.update(librosa_features)
            finally:
                os.remove(audio_file_path)
        
        all_tracks_data.append(track_data)

    # --- Step 3: Fetching Spotify features and merging ---
    print("\nFetching all Spotify audio features in batches...")
    df = pd.DataFrame(all_tracks_data)
    all_audio_features = []
    for i in tqdm(range(0, len(track_ids), 100), desc="Fetching Spotify Features"):
        batch_ids = track_ids[i:i + 100]
        try:
            features_batch = sp.audio_features(batch_ids)
            all_audio_features.extend(features_batch)
        except Exception as e:
            print(f"Error fetching batch starting at index {i}: {e}")
            all_audio_features.extend([None] * len(batch_ids))
        time.sleep(1)

    features_list = [f for f in all_audio_features if f is not None]
    
    if features_list:
        features_df = pd.DataFrame(features_list)
        features_df = features_df.add_prefix('spotify_')
        if 'spotify_id' in features_df.columns:
            final_df = pd.merge(df, features_df, left_on='spotify_id', right_on='spotify_id', how='left')
        else: final_df = df
    else: final_df = df

    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ Success! Your combined profile has been saved to: {output_csv}")
    print(f"Final dataset shape: {final_df.shape}")


if __name__ == '__main__':
    # --- NEW: Set up argument parser ---
    parser = argparse.ArgumentParser(description="Process Spotify liked songs to create an audio profile using yt-dlp.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of songs to process for a test run (e.g., --limit 5). Default is 0 (process all songs)."
    )
    args = parser.parse_args()

    # --- Authentication and fetching liked songs list ---
    print("Connecting to Spotify...")
    auth_manager = SpotifyOAuth(scope="user-library-read", cache_path=".spotipyoauthcache")
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    print("Fetching all your liked songs...")
    all_tracks = []
    offset = 0
    while True:
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        if not results['items']: break
        all_tracks.extend(results['items'])
        offset += 50
    print(f"Found {len(all_tracks)} total liked songs.")

    # --- NEW: Apply the limit if the flag is used ---
    if args.limit > 0:
        print(f"--- Running in test mode: processing only the first {args.limit} songs. ---")
        all_tracks = all_tracks[:args.limit]

    OUTPUT_CSV_PATH = 'spotify_full_audio_profile.csv'
    # Pass the (potentially sliced) list of tracks to the main function
    create_full_audio_profile(all_tracks, OUTPUT_CSV_PATH)