from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import subprocess  # Untuk menjalankan perintah FFmpeg
import threading  # Untuk menjalankan penghapusan otomatis file setelah delay

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

# Fungsi untuk menghapus file setelah delay
def delete_file_after_delay(file_path, delay=3600):
    def delete_file():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} berhasil dihapus setelah {delay} detik.")
        except Exception as e:
            print(f"Gagal menghapus file {file_path}: {e}")

    # Jalankan penghapusan file dalam thread baru
    threading.Timer(delay, delete_file).start()

# Fungsi untuk mengonversi MP3 ke WAV menggunakan FFmpeg
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace('.mp3', '.wav')  # Ubah ekstensi ke .wav
    try:
        # Jalankan perintah FFmpeg untuk konversi
        subprocess.run(['ffmpeg', '-i', mp3_path, wav_path], check=True)
        # Hapus file MP3 setelah berhasil dikonversi
        os.remove(mp3_path)
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting MP3 to WAV: {e}")
        return None
    except OSError as e:
        print(f"Error deleting MP3 file: {e}")
        return None

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

        # Simpan file ke folder yang ditentukan
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Periksa format file
        if file.filename.lower().endswith('.mp3'):
            # Konversi MP3 ke WAV
            file_path_wav = convert_mp3_to_wav(file_path)
            if file_path_wav is None:
                return "Konversi MP3 ke WAV gagal. Pastikan file yang diunggah valid."
            file_path = file_path_wav  # Gunakan file WAV untuk proses berikutnya

        # Ekstraksi fitur dari file yang diunggah
        features = extract_features(file_path)
        if features is None:
            return "Ekstraksi fitur gagal. Coba unggah file lain."

        # Normalisasi fitur menggunakan scaler
        features_scaled = scaler.transform([features])

        # Prediksi genre menggunakan model
        prediction = model.predict(features_scaled)
        predicted_genre = genres[np.argmax(prediction)]

        # Hapus file WAV setelah 30 detik
        delete_file_after_delay(file_path, delay=30)

        # Kembalikan hasil prediksi
        return render_template('index.html', file_path=file_path, prediction=predicted_genre)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)