import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load model machine learning yang sudah dilatih
model = joblib.load('sibi_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Fungsi untuk mengenali gesture berdasarkan landmarks tangan
def recognize_gesture(hand_landmarks):
    # Ambil koordinat x, y, z dari 21 titik landmark
    row = []
    for landmark in hand_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z])
    
    # Prediksi menggunakan model yang sudah dilatih
    prediction = model.predict([np.array(row)])
    return prediction[0] # Mengembalikan hasil terjemahan SIBI (misal: "A", "B", dll)

# Membuka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 1. Ubah BGR ke RGB untuk MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 2. Cek jika tangan terdeteksi di kamera
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Panggil fungsi recognize_gesture dengan mengirim hand_landmarks
            gesture = recognize_gesture(hand_landmarks)    
            
            # Gambar kerangka tangan di frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Tampilkan teks hasil prediksi SIBI di layar
            cv2.putText(frame, f"SIBI: {str(gesture)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame dari webcam    
    cv2.imshow("Hand Gesture Detection", frame)

    # Tekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()