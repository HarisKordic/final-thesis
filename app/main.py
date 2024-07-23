from flask import request, render_template, send_from_directory
import os
from app import app
from utils.image_preprocessing import prepare_image
from utils.gradcam import make_gradcam_heatmap, superimpose_heatmap
from models.model import load_model
import cv2

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img_array = prepare_image(filepath)
        heatmap = make_gradcam_heatmap(img_array, model, 'conv2d_2', ['flatten', 'dense', 'dense_1'])
        superimposed_img = superimpose_heatmap(filepath, heatmap)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
        cv2.imwrite(output_path, superimposed_img)
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
