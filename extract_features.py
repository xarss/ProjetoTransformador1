import os
import shutil
import time
import zipfile
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt

AUDIO_DIR = 'C:/Users/Chico/Dev/PUC/DataScience/data-science/Data/flac/'
PROTOCOL_PATH = 'Data/ASVspoof2021Protocol.txt'
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

    os.makedirs(os.path.join(IMAGE_DIR, 'mfcc'), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DIR, 'lfcc'), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DIR, 'logmel'), exist_ok=True)

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
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

def save_feature_image(feature, file_name, kind):
    path = os.path.join(IMAGE_DIR, kind, f"{file_name}.png")
    plt.figure(figsize=(2, 2))
    plt.imshow(feature, aspect='auto', origin='lower', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_and_save_all_features(y, sr, file_name):
    data = {}

    for kind, extractor in zip(
        ['mfcc', 'lfcc', 'logmel'],
        [extract_mfcc, extract_lfcc, extract_logmel]
    ):
        feature = extractor(y, sr)
        feature_mean = np.mean(feature, axis=1)
        feature_norm = normalize_feature(feature)
        feature_resized = resize(feature_norm, (64, 64), mode='reflect', anti_aliasing=True)
        save_feature_image(feature_resized, file_name, kind)
        data[kind] = feature_mean

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
        df = pd.DataFrame([{
            'file_name': d['file_name'],
            'label': d['label'],
            **{f"{kind}_{i+1}": val for i, val in enumerate(d[kind])}
        } for d in data])
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

num_files = 100
processed_data = []

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
    print("⚠️ Nenhum dado processado.")
