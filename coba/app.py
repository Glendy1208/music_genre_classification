import io
import librosa
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi untuk upload folder (tidak akan digunakan untuk menyimpan file, tapi hanya untuk pemrosesan)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model yang telah dilatih
model = tf.keras.models.load_model('musik_classify2.h5')

# Fungsi untuk mengecek ekstensi file yang diunggah
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk ekstraksi fitur dari file audio (dari memori, bukan dari disk)
def extract_features(file_bytes):
    try:
        # Menggunakan librosa untuk memuat file audio dari bytes
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, duration=30)

        # Ekstraksi fitur audio
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        harmony, perceptr = librosa.effects.harmonic(y=y), librosa.effects.percussive(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Menyusun fitur menjadi array satu dimensi
        features = np.hstack([
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rms), np.var(rms),
            np.mean(spectral_centroid), np.var(spectral_centroid),
            np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
            np.mean(harmony), np.var(harmony),
            np.mean(perceptr), np.var(perceptr),
            tempo,
            np.mean(mfcc, axis=1), np.var(mfcc, axis=1)
        ])

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Fungsi untuk mengklasifikasikan genre berdasarkan fitur yang diekstraksi
def classify_audio(file_bytes):
    features = extract_features(file_bytes)
    if features is None:
        return None

    # Normalisasi fitur menggunakan StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))

    # Prediksi genre menggunakan model ANN
    predictions = model.predict(features_scaled)
    predicted_label = np.argmax(predictions)
    return predicted_label

# Fungsi untuk mengonversi bytes ke Base64
def convert_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode('utf-8')

# Route utama untuk halaman unggah file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Cek apakah file ada dan valid
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        if file.filename == '':
            return "No selected file"
        
        if file and allowed_file(file.filename):
            # Membaca file audio dari memori (tanpa menyimpannya ke disk)
            file_bytes = file.read()
            
            # Klasifikasikan genre
            predicted_label = classify_audio(file_bytes)
            if predicted_label is not None:
                genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                genre = genres[predicted_label]
                
                # Mengonversi file audio ke Base64 untuk pemutaran di HTML
                file_base64 = convert_to_base64(file_bytes)
                
                return render_template('index.html', genre=genre, file_base64=file_base64, filename=file.filename)
            else:
                return "Error in classification"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
