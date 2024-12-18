from flask import render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from app import app
from models.Image_Retrievel import ImageProcessor
import pickle

def serve_image(filename):
    return send_from_directory('/data/images', filename)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
uploaded_file_path = None

@app.route('/')
def index():
    return render_template('index.html', filepath=None, results=[])

@app.route('/upload', methods=['POST'])
def search():
    global uploaded_file_path
    file = request.files['file']
    if file:
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(uploaded_file_path)
        return redirect(url_for('hien_thi_anh'))
    return 'No file uploaded', 400



@app.route('/hien-thi-anh')
def hien_thi_anh():
    global uploaded_file_path, results  # Thêm results là biến toàn cục
    if not uploaded_file_path:
        return redirect(url_for('index'))

    # Khởi tạo ImageProcessor
    image_processor = ImageProcessor()
    dataset_dir = pd.read_csv('data/images.csv')
    src_images = image_processor.load_images(dataset_dir['Links Image'])

    # Tiền xử lý ảnh nguồn
    preprocessed_src_images = np.load('/home/felix/ML1/Project/data/preprocessed_src_images.npy')

    # Tìm ảnh tương tự
    ranked_indices, scores = image_processor.get_similar_images(
        uploaded_file_path,
        preprocessed_src_images,
        src_images,
        dataset_dir,
        top_k=4
    )
    # Lay index cua anh co diem cao nhat
    idx = ranked_indices[0]
    global id
    id = dataset_dir['Index'][idx]

    # Tạo danh sách kết quả
    results = [  # Gán giá trị cho biến toàn cục results
        {
            'image_link': f'static/images/{dataset_dir["Links Image"][idx]}',
            'index': dataset_dir['Index'][idx],
            'score': f'{score:.4f}',
        }
        for idx, score in zip(ranked_indices, scores)
    ]

    return render_template(
        'index.html',
        filepath=uploaded_file_path,
        results=results
    )

list_models = [
    'model_Chaomao.pkl',
    'model_Chichchoe.pkl',
    'model_CuGay.pkl',
    'model_Cong.pkl',
    'model_Cumeo.pkl',
    'model_Catnho.pkl',
    'model_Yenui.pkl'
]
id = 0
MODEL_PATH = 'naof/'



@app.route('/Du-doan')
def du_doan():
    global id, uploaded_file_path, results  # Ensure results is declared global

    try:
        # Load model
        model_dir = os.path.join(MODEL_PATH, list_models[id])
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)
        # Get input from form
        day = request.args.get('day', '1')
        if not day.isdigit() or int(day) <= 0:
            return "Invalid day input", 400

        day = int(day)

        # Make prediction
        future = model.make_future_dataframe(periods=day)
        prediction = model.predict(future)

        # Plot and save prediction image
        global prediction_img
        prediction_img = 'static/prediction.png'
        model.plot(prediction)

        plt.savefig(prediction_img)

        # Determine risk prediction
        list_predict = ['Không có nguy cơ', 'Nguy cơ thấp', 'Nguy cơ trung bình', 'Nguy cơ cao']
        if id == 4 or id == 3:
            predit_RF = f'Nguy cơ của loài chim này là :  {list_predict[3]}'
        elif id == 2:
            predit_RF = f'Nguy cơ của loài chim này là :  {list_predict[2]}'
        elif id == 5 or id == 6:
            predit_RF = f'Nguy cơ của loài chim này là :  {list_predict[2]}'
        elif id == 1:
            predit_RF = f'Nguy cơ của loài chim này là :  {list_predict[0]}'

        # Render template with all variables
        return render_template(
            'index.html',
            filepath=uploaded_file_path,
            results=results,  # Use results from global variable
            prediction=prediction_img,
            predit_RF=predit_RF  # Pass predit_RF to the template
        )
    except FileNotFoundError:
        return "Model file not found", 500
    except Exception as e:
        return f"Error: {str(e)}", 500

