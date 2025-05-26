import os
import shutil
import zipfile
import argparse
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Extract and save audio features and images.")
parser.add_argument('--flac_dir', required=True, help='Path to the folder with .flac audio files')
parser.add_argument('--protocol_file', required=True, help='Path to the ASVspoof protocol .txt file')
args = parser.parse_args()

AUDIO_DIR = args.flac_dir
PROTOCOL_PATH = args.protocol_file
OUTPUT_DIR = 'Data'
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'Images')
ZIP_PATH = os.path.join(OUTPUT_DIR, 'images.zip')
CSV_PATHS = {
    'mfcc': os.path.join(OUTPUT_DIR, 'features_mfcc.csv'),
    'lfcc': os.path.join(OUTPUT_DIR, 'features_lfcc.csv'),
    'logmel': os.path.join(OUTPUT_DIR, 'features_logmel.csv')
}

def clean_output():
    print("Removing old files")
    if os.path.exists(IMAGE_DIR):
        shutil.rmtree(IMAGE_DIR)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    for path in CSV_PATHS.values():
        if os.path.exists(path):
            os.remove(path)

    for kind in ['mfcc', 'lfcc', 'logmel']:
        os.makedirs(os.path.join(IMAGE_DIR, kind), exist_ok=True)

def extract_mfcc(y, sr, n_mfcc=40, hop_length=256):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

def extract_logmel(y, sr, n_mels=128, hop_length=256):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(mel)

def extract_lfcc(y, sr, n_lfcc=40, hop_length=256):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    power_db = librosa.power_to_db(S)
    return librosa.feature.mfcc(S=power_db, sr=sr, n_mfcc=n_lfcc)

def normalize_feature(x):
    norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)
    if np.max(norm) < 0.05:
        print("⚠️ Warning: low energy spectrogram.")
    return norm

def save_feature_image(feature, file_name, kind, sr=16000, hop_length=256):
    path = os.path.join(IMAGE_DIR, kind, f"{file_name}.png")
    plt.figure(figsize=(8, 4), dpi=200)
    librosa.display.specshow(feature, sr=sr, hop_length=hop_length,
                              x_axis='time',
                              y_axis='mel' if kind == 'logmel' else 'linear',
                              cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_and_save_all_features(y, sr, file_name):
    data = {}
    for kind, extractor in zip(['mfcc', 'lfcc', 'logmel'], [extract_mfcc, extract_lfcc, extract_logmel]):
        try:
            feature = extractor(y, sr)
            feature_mean = np.mean(feature, axis=1)
            feature_norm = normalize_feature(feature)
            save_feature_image(feature_norm, file_name, kind, sr=sr)
            data[kind] = feature_mean
        except Exception as e:
            print(f"⚠️ Error processing {kind} for {file_name}: {e}")
            data[kind] = np.zeros(40)
    return data

def process_audio_row(row):
    file_path = os.path.join(AUDIO_DIR, row['file_name'] + '.flac')
    if not os.path.exists(file_path):
        print(f"Not found: {file_path}")
        return None

    try:
        y, sr = librosa.load(file_path, sr=None)
        features = extract_and_save_all_features(y, sr, row['file_name'])
        return {
            'file_name': row['file_name'],
            'label': 1 if row['label'] == 'bonafide' else 0,
            'mfcc': features['mfcc'],
            'lfcc': features['lfcc'],
            'logmel': features['logmel']
        }
    except Exception as e:
        print(f"Error: {file_path} - {e}")
        return None

def zip_and_delete_images():
    print("Compressing images...")
    with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
        for root, _, files in os.walk(IMAGE_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, IMAGE_DIR)
                zipf.write(filepath, arcname)
    shutil.rmtree(IMAGE_DIR)
    print(f"Images compressed to file '{ZIP_PATH}'")

def save_feature_csvs(data):
    for kind in ['mfcc', 'lfcc', 'logmel']:
        rows = []
        for d in data:
            rows.append({
                'file_name': d['file_name'],
                'label': d['label'],
                **{f"{kind}_{i+1}": val for i, val in enumerate(d[kind])}
            })
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATHS[kind], index=False)
        print(f"Saved '{kind}' features in file '{CSV_PATHS[kind]}'")

# Execução principal
clean_output()

protocol = pd.read_csv(PROTOCOL_PATH, sep=' ', header=None)
protocol.columns = [
    "id", "file_name", "codec", "source_db", "system_id", "label",
    "trim_status", "set_type", "spoof_category",
    "track", "team", "subset", "group"
]

num_files = 2000
processed_data = []

os.makedirs(OUTPUT_DIR, exist_ok=True)
for _, row in tqdm(protocol.iterrows(), total=num_files, desc="Extracting features"):
    if len(processed_data) >= num_files:
        break
    result = process_audio_row(row)
    if result:
        processed_data.append(result)

if processed_data:
    save_feature_csvs(processed_data)
    zip_and_delete_images()
else:
    print("No data was processed")
