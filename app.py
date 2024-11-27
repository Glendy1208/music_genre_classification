from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Path folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan scaler
model = tf.keras.models.load_model('model_code1.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Label genre musik
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Fungsi untuk ekstraksi fitur dari file musik
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30, sr=22050)

        # Ekstraksi fitur
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Menggabungkan semua fitur
        features = np.hstack([mfccs, chroma, spectral_contrast, zero_crossings, tempo])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Periksa format file
        if file and file.filename.endswith(('.wav', '.mp3')):
            # Simpan file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Ekstraksi fitur dari file yang diunggah
            features = extract_features(file_path)
            if features is None:
                return "Ekstraksi fitur gagal. Coba unggah file lain."

            # Normalisasi fitur menggunakan scaler
            features_scaled = scaler.transform([features])

            # Prediksi genre menggunakan model
            prediction = model.predict(features_scaled)
            predicted_genre = genres[np.argmax(prediction)]

            # Kembalikan hasil prediksi
            return render_template('index.html', file_path=file_path, prediction=predicted_genre)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
