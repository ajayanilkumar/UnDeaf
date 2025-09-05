import pandas as pd
import librosa
import numpy as np
import os
from tqdm import tqdm

def extract_features(file_path):
    """
    Extracts a comprehensive set of musical features from an audio file.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        dict: A dictionary containing the extracted features. Returns None if the file cannot be processed.
    """
    try:
        # Load audio file, using a 30-second duration for consistency
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        features = {}

        # 1. Timbre Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
            features[f'mfcc{i}_std'] = np.std(mfccs[i-1])

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_std'] = np.std(spec_cent)

        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
        features['spectral_rolloff_std'] = np.std(spec_rolloff)

        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)

        # 2. Rhythm & Tempo Features
        # Use aggregate=None to handle potential cases where tempo is not stable
        tempo_values = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        features['tempo'] = np.mean(tempo_values) if tempo_values.size > 0 else 0

        # 3. Dynamics & Energy Features
        rms = librosa.feature.rms(y=y)
        features['rms_energy_mean'] = np.mean(rms)
        features['rms_energy_std'] = np.std(rms)

        # 4. Pitch & Harmony (Tonal Features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_feature_dataset(data_dir, output_csv):
    """
    Processes all audio files in a directory, extracts features, and saves them to a CSV.

    Args:
        data_dir (str): The path to the root directory of the music dataset.
        output_csv (str): The path to save the output CSV file.
    """
    all_features = []
    
    # Using tqdm for a progress bar
    # The outer loop iterates through genres (e.g., 'blues', 'rock')
    for genre_folder in tqdm(os.listdir(data_dir), desc="Processing Genres"):
        genre_path = os.path.join(data_dir, genre_folder)
        if os.path.isdir(genre_path):
            # The inner loop iterates through songs in each genre folder
            for filename in os.listdir(genre_path):
                if filename.endswith(('.wav', '.mp3', '.au')):
                    file_path = os.path.join(genre_path, filename)
                    
                    # Extract features for one song
                    features = extract_features(file_path)
                    
                    if features:
                        # Add metadata
                        features['filename'] = filename
                        features['genre'] = genre_folder
                        # You can add artist info if your file naming convention supports it
                        # features['artist'] = filename.split('-')[0] 
                        all_features.append(features)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns to have metadata first
    metadata_cols = ['filename', 'genre']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully created dataset at: {output_csv}")
    print(f"Dataset shape: {df.shape}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # IMPORTANT: Update this path to your dataset's location
    MUSIC_DATASET_PATH = '/Users/ajay.kumar/Downloads/genres'
    OUTPUT_CSV_PATH = 'musical_features_dataset.csv'
    
    # Check if the path exists before running
    if not os.path.exists(MUSIC_DATASET_PATH):
        print(f"Error: The specified directory does not exist: {MUSIC_DATASET_PATH}")
        print("Please download the GTZAN dataset and update the 'MUSIC_DATASET_PATH' variable.")
    else:
        create_feature_dataset(MUSIC_DATASET_PATH, OUTPUT_CSV_PATH)