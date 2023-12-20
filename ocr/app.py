# Import modul yang diperlukan
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from flask import Flask, jsonify, request
from functions import *
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# import for ocr



# Definisikan variabel global
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# Buat fungsi untuk memeriksa apakah ekstensi file diperbolehkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Buat objek Flask
app = Flask(__name__)

# Buat rute untuk mengunggah media
@app.route('/media/upload', methods=['POST'])
def upload_media():

    method = "manual"
    
    if request.args.get('method') is not None:
        method = request.args.get('method')

    if 'image' not in request.files:
        return jsonify({'error': 'media not provided'}), 400

    # Ambil file yang diunggah
    file = request.files['image']
    image = request.files['image'].stream
    img = cv2.imdecode((np.fromstring(image.read(), np.uint8)), cv2.IMREAD_COLOR)
    # print(f"isi file:{img}")


    # Cek apakah file kosong
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    # Cek apakah ekstensi file diperbolehkan
    if file and not allowed_file(file.filename):
        return jsonify({'error': 'file extension not allowed'}), 400

    # Amankan nama file
    filename = secure_filename(file.filename)

    image = img

    crop =[]
    sorted_contours_lines = None
#region lokalisasi
    

    try:
        clustered_image = cluster_image(image)
    
        # grayscale
        clustered_image = clustered_image.astype(np.uint8)
    
        # Option 1: Using OpenCV cvtColor
        gray = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY)
    
        # contour detection
        image_copy_contours, contours = contour_detection(gray, image)
    
        # largest contour
        largest_contour, image =  find_largest_contour(contours,image)
    
        # black otside
        black_outside = make_outside_black(image,largest_contour)
    
        # Straightened
        ktp_width, ktp_height = 856, 540

        crop = get_transformed_image(black_outside,contours)
    
        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # cv2.imshow("",crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        h, w, c = crop.shape
        
        if w > 1000:
        
            new_w = 1000
            ar = w/h
            new_h = int(new_w/ar)
        
            crop = cv2.resize(crop, (new_w, new_h), interpolation = cv2.INTER_AREA)
    
        clustered_image = cluster_image(crop)
        clustered_image = clustered_image.astype(np.uint8)
        
        gray = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY)
        
        # Fungsi inversi warna
        inverted_img = get_inverted_image(gray)
    
        #dilation
        kernel = np.ones((3,25), np.uint8)
        dilated = cv2.dilate(inverted_img, kernel, iterations = 1)
    
        # find contours
        (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("menjalankan sorted_contours_lines")
        sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)
    
        # get text locations in contours
        img2 = get_sorted_contours_lines(sorted_contours_lines, crop)
    
            # List untuk menyimpan hasil crop
        crop_results = save_contoured_area(crop,sorted_contours_lines)

    except Exception as e:
    # Handle other types of exceptions
        print(f"An error occurred: {e} in region localisasi")
#endregion
    
#region ocr
    
    try:
        # Load model
        network = load_model('models/my_model.h5')
        
        # List untuk menyimpan hasil crop
        crop_results = []
        
        # Variabel untuk menyimpan hasil deteksi karakter
        detected_words = []

        # Loop untuk menampilkan setiap hasil crop satu per satu
        for ctr in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
        
            # Periksa ukuran bounding box
            if (w >= 10 and w <= 1000) and (h >= 20 and h <= 50):
                # Crop bagian gambar yang ada di dalam bounding box
                cropped_img = crop[y:y+h, x:x+w]
        
                # Konversi ke citra greyscale
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
                # Thresholding dengan Otsu
                _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
                # Invertion
                invertion = 255 - otsu
        
                # Tambahkan hasil crop yang telah diproses ke dalam list
                crop_results.append(invertion)
        
                # Display crop_result
        
                # Segmentation of letters
                contours_letter, _ = cv2.findContours(invertion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_letter = sorted(contours_letter, key=lambda x: cv2.boundingRect(x)[0])
        
                # Array untuk menyimpan hasil prediksi huruf pada satu crop
                predicted_chars_crop = []
        
                # Segmentasi huruf berdasarkan bounding box dari tiap kontur
                for contour in contours_letter:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
        
                    # Tambahkan huruf ke list jika memenuhi kriteria tertentu (misal: min/max luas)
                    if cv2.contourArea(contour) > 6 and w1/h1 < 2:
                        letter = otsu[y1:y1+h1, x1:x1+w1]
        
                        # Resize gambar huruf agar sesuai dengan bentuk input model
                        letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_AREA)
        
                        # Convert the image to PIL format
                        pil_img = Image.fromarray(letter)
        
                        # # Tentukan ukuran padding (misalnya, 5 piksel)
                        # padding_size = 2
        
                        # # Buat border padding menggunakan ImageOps.expand
                        # padded_letter = ImageOps.expand(pil_img, border=padding_size, fill=(0,))
        
                        # Konversi citra PIL ke array NumPy
                        numpy_letter = np.array(pil_img)
        
                        # Display the resized and padded segmented letter
    
        
                        # Konversi citra PIL ke array NumPy
                        numpy_letter_resized = np.array(pil_img)
        
                        # Menambahkan dimensi baru pada axis terakhir
                        numpy_letter_resized_expanded = np.expand_dims(numpy_letter_resized, axis=-1)
        
                        # Menambahkan dimensi batch (None) pada axis pertama
                        numpy_letter_resized_expanded_batched = np.expand_dims(numpy_letter_resized_expanded, axis=0)
        
                        # Cetak bentuk numpy_letter_resized_expanded_batched
                        print(f"Shape of numpy_letter_resized_expanded_batched: {numpy_letter_resized_expanded_batched.shape}")
        
                        # Lakukan prediksi menggunakan model
                        if network is not None:
                            prediction = network.predict(numpy_letter_resized_expanded_batched)
                            char_detected = chr(ord('A') + np.argmax(prediction))  # Misalnya, anggap model output sebagai one-hot encoding
        
                            # Menambahkan teks hasil prediksi pada gambar
                            cv2.putText(numpy_letter_resized, char_detected, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            # Cetak hasil prediksi
                            print(f"Hasil prediksi: {char_detected}")
        
                            # Tambahkan hasil prediksi ke dalam list
                            predicted_chars_crop.append(char_detected)
        
                # Gabungkan hasil prediksi huruf pada satu crop menjadi satu string
                detected_word = ''.join(predicted_chars_crop)
                detected_words.append(detected_word)
        
        # Menampilkan hasil deteksi karakter untuk setiap crop
        for i, detected_word in enumerate(detected_words):
            print(f"Detected Word for Crop {i + 1}: {detected_word}")

    except Exception as e:
    # Handle other types of exceptions
        print(f"An error occurred: {e} in region ocr")
    
#endregion

#region tesseract

    try:
    
# List untuk menyimpan hasil crop
        crop_results = []
    
    # Variabel untuk menyimpan hasil deteksi karakter
        detected_words = []

        # Loop untuk menampilkan setiap hasil crop satu per satu
        for ctr in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
        
            # Periksa ukuran bounding box
            if (w >= 10 and w <= 1000) and (h >= 20 and h <= 50):
                # Crop bagian gambar yang ada di dalam bounding box
                cropped_img = crop[y:y+h, x:x+w]
        
                # Konversi ke citra greyscale
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
                # Thresholding dengan Otsu
                _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
                # Invertion
                invertion = 255 - otsu
        
                # Tambahkan hasil crop yang telah diproses ke dalam list
                crop_results.append(invertion)
        
                # Display crop_results
                
        
                # Segmentation of letters
                contours_letter, _ = cv2.findContours(invertion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_letter = sorted(contours_letter, key=lambda x: cv2.boundingRect(x)[0])
        
                # Array untuk menyimpan hasil prediksi huruf pada satu crop
                predicted_chars_crop = []
        
                # Segmentasi huruf berdasarkan bounding box dari tiap kontur
                for contour in contours_letter:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
        
                    # Tambahkan huruf ke list jika memenuhi kriteria tertentu (misal: min/max luas)
                    if cv2.contourArea(contour) > 6 and w1/h1 < 2:
                        letter = invertion[y1:y1+h1, x1:x1+w1]
        
                        # Convert the image to PIL format
                        pil_img = Image.fromarray(letter)
        
                        # Tentukan ukuran padding (misalnya, 5 piksel)
                        padding_size = 10
        
                        # Buat border padding menggunakan ImageOps.expand
                        padded_letter = ImageOps.expand(pil_img, border=padding_size, fill=(0,))
        
                        # Konversi citra PIL ke array NumPy
                        numpy_letter = np.array(padded_letter)
        
                        # Resize pelebaran width
                        desired_width = 30  # Sesuaikan dengan lebar yang diinginkan
                        current_width = numpy_letter.shape[1]
                        scale_factor = desired_width / current_width
                        numpy_letter_resized = cv2.resize(numpy_letter, (desired_width, int(numpy_letter.shape[0] * scale_factor)))
        
                        # Display the resized and padded segmented letter
                        
        
                        # Prediksi huruf
                        predicted_char = pytesseract.image_to_string(numpy_letter_resized, config='--psm 10')
                        print(f"Predicted text for letter: {predicted_char}")
        
                        # Tambahkan hasil prediksi ke dalam list
                        predicted_chars_crop.append(predicted_char)
        
                # Gabungkan hasil prediksi huruf pada satu crop menjadi satu string
                detected_word = ''.join(predicted_chars_crop)
                detected_words.append(detected_word)
        
        # Menampilkan hasil deteksi karakter untuk setiap crop
        for i, detected_word in enumerate(detected_words):
            print(f"Detected Word for Crop {i + 1}: {detected_word}")

    except Exception as e:
        # Handle other types of exceptions 
        print(f"An error occurred: {e} in region tesseract")
#endregion


#     clustered_image = clustered_image.astype(np.uint8)

# # Option 1: Using OpenCV cvtColor
# gray = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY)


    print(f"detected_words{detected_words}")
    # Kembalikan respons sukses
    return jsonify({'detected_words': detected_words})

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True, port=5006)