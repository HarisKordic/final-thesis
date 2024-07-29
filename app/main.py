from flask import request, render_template, send_from_directory, url_for
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
        
        # Preprocess the image and generate heatmap
        img_array = prepare_image(filepath)
        heatmap = make_gradcam_heatmap(img_array, model, 'conv2d_2', ['flatten', 'dense', 'dense_1'])
        superimposed_img = superimpose_heatmap(filepath, heatmap)
        
        # Save the superimposed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
        cv2.imwrite(output_path, superimposed_img)
         # Manually construct the URLs for images
        original_image_url = f'http://localhost:5000/uploads/{filename}'
        heatmap_image_url = f'http://localhost:5000/uploads/gradcam_{filename}'

        print(f"Original image URL: {original_image_url}")
        print(f"Heatmap image URL: {heatmap_image_url}")

        
        # Return the template with both images
        return render_template('display_images.html', 
                               original_image=original_image_url, 
                               heatmap_image=heatmap_image_url)

@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(f"../{app.config['UPLOAD_FOLDER']}", filename);


