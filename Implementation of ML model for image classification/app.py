import os
import logging
from flask import Flask, request, render_template
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load models
mobilenetv2_model = MobileNetV2(weights='imagenet')
gender_model = load_model('models/gender_model.h5')

# Configure logging for misclassifications
logging.basicConfig(filename='misclassifications.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Image upload and classification route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(file_path)

        # Get selected model
        model_choice = request.form.get('model_choice')
        actual_gender = request.form.get('actual_gender')  # Optional field for actual gender

        if model_choice == 'mobilenetv2':
            # Process image for MobileNetV2
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict using MobileNetV2
            predictions = mobilenetv2_model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=1)
            result_label = decoded_predictions[0][0][1]
            result_confidence = decoded_predictions[0][0][2] * 100  # Convert to percentage

        elif model_choice == 'gender':
            # Process image for gender model
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize for gender model

            # Predict using gender model
            predictions = gender_model.predict(img_array)

            male_confidence = predictions[0][0]  # Confidence for male class
            female_confidence = 1 - male_confidence  # Confidence for female class

            # Log the raw predictions for debugging
            logging.info(f"Predictions for image: {file_path}, Male: {male_confidence}, Female: {female_confidence}")

            # Threshold adjusted to 0.6
            if male_confidence > 0.6:
                result_label = 'Male'
                result_confidence = male_confidence * 100  # Convert to percentage
            elif female_confidence > 0.6:
                result_label = 'Female'
                result_confidence = female_confidence * 100  # Convert to percentage
            else:
                result_label = 'Uncertain'
                result_confidence = max(male_confidence, female_confidence) * 100  # Display the higher confidence

            # Log misclassifications if actual gender is provided
            if actual_gender:
                if actual_gender != result_label:
                    logging.info(f"Misclassified Image: {file_path}, Predicted: {result_label}, Actual: {actual_gender}, Confidence: {result_confidence:.2f}%")

        else:
            return render_template('index.html', error='Invalid model choice')

        # Ensure result_confidence is a scalar value
        result_confidence = float(result_confidence)

        # Render result
        return render_template('result.html', result=result_label, confidence=f"{result_confidence:.2f}%", image_path=file_path)

    return render_template('index.html', error='Invalid file type or error during upload')


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
