import pyrebase
import requests
from scipy import signal
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import time


config = {
    'apiKey': "AIzaSyBfwJoBt2kT0iOMjlDBw_heFaqjwjlp5ZU",
    'authDomain': "medical-record-7557a.firebaseapp.com",
    'databaseURL': "https://medical-record-7557a-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "medical-record-7557a",
    'storageBucket': "medical-record-7557a.appspot.com",
    'messagingSenderId': "973084416066",
    'appId': "1:973084416066:web:50c8c2831db284a7e835db",
    'measurementId': "G-MZ9NN8VEQZ"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

def stream_handler(message):
    if message["event"] == "put":
        # Mendapatkan data terbaru
        data = message["data"]
        if isinstance(data, dict):
            # Mendapatkan informasi file terbaru
            latest_file_info = get_latest_file_info(data)
            # Mendownload file terbaru
            download_latest_file(latest_file_info)
            #prediksi sbp dan dbp
            predictions_DBP, predictions_SBP = predict("processed_data.csv")

            update_latest_key(latest_file_info, predictions_DBP, predictions_SBP)

            latest_file_info = get_latest_file_info(data)

            data_classification = preprocess_classification_data(latest_file_info)
            klasifikasi_hipertensi = klasifikasi("classification_input.xlsx")

            update_classification(latest_file_info, klasifikasi_hipertensi)

# Mendapatkan informasi file terbaru dari data
def get_latest_file_info(data):
    if isinstance(data, dict):
        sorted_data = sorted(data.items(), key=lambda x: x[1].get("timestamp", 0))
        latest_key, latest_file_info = sorted_data[-1]
        if isinstance(latest_file_info, dict):
            latest_file_info["key"] = latest_key
            print("Data terbaru:")
            print("", latest_file_info)
            print("Key:", latest_key)
            return latest_file_info
        else:
            print("Informasi file terbaru tidak valid.")
    else:
        print("Data tidak valid.")

# Pre-Processing
def preprocess_data(input_file, output_file):
    fs = 1000  # Sampling rate (Hz)
    lowcut = 0.5  # Lower cutoff frequency (Hz)
    highcut = 25  # Upper cutoff frequency (Hz)
    order = 4  # Filter order

    # Calculate Nyquist frequency
    nyquist = 0.5 * fs

    # Calculate filter frequencies
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create Butterworth filter coefficients
    b, a = signal.butter(order, [low, high], btype='band')

    window_size = 50

    # Load the input data
    ppg_signal = np.loadtxt(input_file)

    # Pad the signal with edge values for moving average calculation
    padding = (window_size - 1) // 2
    ppg_signal_padded = np.pad(ppg_signal, (padding, padding), mode='edge')
    moving_avg = np.convolve(ppg_signal_padded, np.ones(window_size) / window_size, mode='valid')

    filtered_ppg = signal.filtfilt(b, a, moving_avg)

    filtered_ppg = filtered_ppg.astype(np.float32)
    # Truncate filtered_ppg to 2100 values
    filtered_ppg = filtered_ppg[:2099]

    # Create a DataFrame with filtered_ppg as the data
    df = pd.DataFrame(data=filtered_ppg).T

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print("Data berhasil diproses dan disimpan:", output_file)

def predict(input_file):
    #prediksi SBP dan DBP
    model_DBP = load_model('DBP.h5', compile=False)
    model_SBP = load_model('SBP.h5', compile=False)

    # Load the preprocessed data
    preprocessed_data = pd.read_csv(input_file)

    # Perform prediction using the model
    predictions_DBP = model_DBP.predict(preprocessed_data)
    predictions_SBP = model_SBP.predict(preprocessed_data)

    print("DBP predictions:", predictions_DBP)
    print("SBP predictions:", predictions_SBP)

    return predictions_DBP, predictions_SBP

def update_latest_key(file_info, predictions_DBP, predictions_SBP):
    latest_key = file_info["key"]
    dbp_prediction = predictions_DBP[0]  # Ambil elemen pertama dari array ndarray
    sbp_prediction = predictions_SBP[0]  # Ambil elemen pertama dari array ndarray

    # Ubah hasil prediksi menjadi tipe data yang dapat di-serialisasi menjadi JSON
    dbp_prediction = float(dbp_prediction)
    sbp_prediction = float(sbp_prediction)

    # Update nilai DBP dan SBP pada database Firebase
    db.child("records").child(latest_key).update({
        "DBP": dbp_prediction,
        "SBP": sbp_prediction
    })
    # Gabungkan informasi DBP dan SBP ke dalam latest_file_info
    file_info["DBP"] = dbp_prediction
    file_info["SBP"] = sbp_prediction

    return file_info

def preprocess_classification_data(file_info):
    # Load data for classification
    # Menggabungkan data input dengan nilai prediksi DBP dan SBP

    df = pd.DataFrame(file_info, index=[0])

    # Simpan data input dan prediksi dalam file CSV
    df.to_excel("classification_input.xlsx", index=False)

    data_classification = pd.read_excel("classification_input.xlsx")
    print("Data sebelum diubah urutan kolom:")
    print(data_classification)

    # Mengubah urutan kolom
    new_column_order = ['sex', 'umur', 'height', 'weight', 'heartRate', 'bmi', 'DBP', 'SBP', 'file', 'nama', 'noPasien', 'key']
    data_classification = data_classification.reindex(columns=new_column_order)

    # Replace 'M' with 1 and 'F' with 0 in the 'sex' column
    data_classification['sex'] = data_classification['sex'].map({'M': 1, 'F': 0}).fillna(data_classification['sex'])

    # Remove unnecessary columns
    data_classification = data_classification.drop(['file', 'nama', 'noPasien', 'key'], axis=1)

    # Save DataFrame to CSV file
    data_classification.to_excel("classification_input.xlsx", index=False)
    print("Data setelah diubah urutan kolom:")
    print(data_classification)

    return data_classification


def klasifikasi(input_file):
    #Klasifikasi
    model_classification = tf.keras.models.load_model('klasifikasi.h5', compile=False)

    # Load the preprocessed data
    data_classification = pd.read_excel("classification_input.xlsx")

    # Perform prediction using the model
    klasifikasi_hipertensi = model_classification.predict(data_classification)

    # Mengambil indeks dengan probabilitas terbesar
    klasifikasi_hipertensi = np.argmax(klasifikasi_hipertensi, axis=1)

    print("Hasil Klasifikasi:", klasifikasi_hipertensi)

    return klasifikasi_hipertensi


def update_classification(file_info, klasifikasi_result):
    latest_key = file_info["key"]

    # Membaca hasil klasifikasi
    klasifikasi_label = {
        0: "Pra Hipertensi",
        1: "Normal",
        2: "Hipertensi Tingkat 1",
        3: "Hipertensi Tingkat 2"
    }
    klasifikasi_result = klasifikasi_result[0]  # Ambil elemen pertama dari array ndarray
    klasifikasi = klasifikasi_label[klasifikasi_result]

    # Mengupdate nilai klasifikasi pada database Firebase
    db.child("records").child(latest_key).update({
        "Diagnosis": klasifikasi
    })

    print("Klasifikasi berhasil diperbarui:", klasifikasi)


# Mendownload file terbaru
def download_latest_file(file_info):
    file_url = file_info["file"]
    response = requests.get(file_url)
    with open("downloaded_file.txt", "wb") as file:
        file.write(response.content)
    print("File terbaru berhasil diunduh")
    
    # Memanggil fungsi pre-processing
    preprocess_data("downloaded_file.txt", "processed_data.csv")

# Loop untuk terus mendengarkan perubahan data dan memprosesnya
#my_stream = db.child("records").stream(stream_handler)
while True:
    my_stream = db.child("records").stream(stream_handler)
    time.sleep(10)

