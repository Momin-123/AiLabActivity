from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import pickle

app = Flask(__name__)

try:
    model = pickle.load(open('model/fruit_model.pkl', 'rb'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: Could not load model. Make sure 'fruit_model.pkl' exists in the model directory. Error: {str(e)}")
    model = None
CLASS_NAMES = ['apple', 'banana', 'orange'] 

def preprocess_image(image_array):
    img = cv2.resize(image_array, (224, 224))
    if len(img.shape) == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img_flat = img.reshape(1, -1)  
    return img_flat

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        processed_image = preprocess_image(image)

        if model is not None:
            predictions = model.predict_proba(processed_image)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': f'{confidence:.2%}'
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 