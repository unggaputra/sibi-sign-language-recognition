import os
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Direktori dataset Roboflow Anda
base_dir = r'E:\Project\image-processing-portofolio\SIBI.v2-sibi.tensorflow'

# Fungsi untuk memproses dataset dari folder (train/valid/test)
def extract_features_from_roboflow(split_name):
    csv_path = os.path.join(base_dir, split_name, '_annotations.csv')
    img_dir = os.path.join(base_dir, split_name)
    
    if not os.path.exists(csv_path):
        print(f"File anotasi tidak ditemukan di: {csv_path}")
        return [], []

    df = pd.read_csv(csv_path)
    data = []
    labels = []

    print(f"Memproses dataset dari folder: {split_name}...")
    
    for index, row in df.iterrows():
        filename = row['filename']
        class_name = row['class']
        
        # Buka gambar
        img_path = os.path.join(img_dir, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Jika tangan terdeteksi oleh MediaPipe
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_row = []
                for landmark in hand_landmarks.landmark:
                    landmarks_row.extend([landmark.x, landmark.y, landmark.z])
                
                data.append(landmarks_row)
                labels.append(class_name)
                
    return data, labels

# 1. Kumpulkan data dari folder train dan valid agar model lebih pintar
X_train_list, y_train_list = extract_features_from_roboflow('train')
X_valid_list, y_valid_list = extract_features_from_roboflow('valid')

# Gabungkan data train dan valid
X_data = X_train_list + X_valid_list
y_data = y_train_list + y_valid_list

if len(X_data) > 0:
    X = np.array(X_data)
    y = np.array(y_data)

    # 2. Bagi data untuk training dan testing (80% train, 20% test untuk evaluasi)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Latih model Random Forest
    print("Melatih model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluasi akurasi model
    accuracy = model.score(X_test, y_test)
    print(f"Akurasi Model: {accuracy * 100:.2f}%")

    # 5. Simpan model menjadi file pkl
    joblib.dump(model, 'sibi_model.pkl')
    print("Model berhasil disimpan sebagai sibi_model.pkl")
else:
    print("Tidak ada data tangan yang berhasil diekstrak. Pastikan gambar di dalam folder jelas dan tangan terlihat oleh MediaPipe.")