import os
import zipfile
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

# Step 1: Define paths and URLs
OUTPUT_DIR = 'Data'
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'Images')
CSV_URLS = {
    'mfcc': "https://github.com/xarss/ProjetoTransformador1/raw/main/Data/features_mfcc.csv",
    'lfcc': "https://github.com/xarss/ProjetoTransformador1/raw/main/Data/features_lfcc.csv",
    'logmel': "https://github.com/xarss/ProjetoTransformador1/raw/main/Data/features_logmel.csv"
}
ZIP_URL = "https://github.com/xarss/ProjetoTransformador1/raw/main/Data/images.zip"

# Step 2: Create Data folder and download files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download CSV files
for kind, url in CSV_URLS.items():
    csv_path = os.path.join(OUTPUT_DIR, f"features_{kind}.csv")
    if not os.path.exists(csv_path):
        print(f"Downloading {kind} features...")
        response = requests.get(url)
        with open(csv_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {kind} features.")

# Download ZIP file with images
zip_path = os.path.join(OUTPUT_DIR, 'images.zip')
if not os.path.exists(zip_path):
    print("Downloading images.zip...")
    response = requests.get(ZIP_URL)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print("Downloaded images.zip.")

# Step 3: Unzip images
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(IMAGE_DIR)
    print("Images unzipped.")

# Step 4: Load features into a pandas DataFrame
mfcc_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'features_mfcc.csv'))
lfcc_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'features_lfcc.csv'))
logmel_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'features_logmel.csv'))

# Merge all features into a single DataFrame
features_df = pd.merge(mfcc_df, lfcc_df, on='file_name', suffixes=('_mfcc', '_lfcc'))
features_df = pd.merge(features_df, logmel_df, on='file_name')

# Step 5: Split data into training, validation, and testing sets
X = features_df.drop(columns=['file_name', 'label'])
y = features_df['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 6: SVM and Random Forest
# SVM Classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# Step 7: Shallow CNN on Images
# Load images for CNN
def load_image(file_path, img_size=(128, 128)):
    img = image.load_img(file_path, target_size=img_size, color_mode='rgb')
    img_array = image.img_to_array(img) / 255.0
    return img_array

# Prepare image data
image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
X_img = np.array([load_image(file) for file in image_files])
y_img = [int(file.split('/')[-2] == 'bonafide') for file in image_files]  # Assuming folder structure 'bonafide' or 'spoof'

# Split image data into train, validation, test
X_train_img, X_temp_img, y_train_img, y_temp_img = train_test_split(X_img, y_img, test_size=0.3, random_state=42)
X_val_img, X_test_img, y_val_img, y_test_img = train_test_split(X_temp_img, y_temp_img, test_size=0.5, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_img, y_train_img, validation_data=(X_val_img, y_val_img), epochs=5, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_img, y_test_img)
print(f"Shallow CNN Test Accuracy: {test_acc * 100:.2f}%")
