from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import glob
import time
import shutil
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained YOLO model
model = YOLO('runs/segment/train2/weights/best.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_latest_predict_folder():
    # Find all predict folders
    predict_folders = glob.glob('runs/segment/predict*')
    if not predict_folders:
        logger.error("No predict folders found")
        return None
    
    # Get the latest folder by number
    latest_folder = None
    latest_number = -1
    
    for folder in predict_folders:
        try:
            # Extract the number from the folder name (e.g., 'predict13' -> 13)
            folder_number = int(folder.split('predict')[-1])
            if folder_number > latest_number:
                latest_number = folder_number
                latest_folder = folder
        except ValueError:
            continue
    
    if latest_folder:
        logger.debug(f"Latest predict folder: {latest_folder}")
        return latest_folder
    else:
        logger.error("Could not find a valid prediction folder")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Secure the filename and save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.debug(f"Saved uploaded file to: {filepath}")
            
            # Perform prediction
            results = model.predict(filepath, save=True)
            logger.debug("Prediction completed")
            
            # Wait a moment for the file to be fully saved
            time.sleep(2)  # Increased wait time
            
            # Get the latest prediction folder
            latest_predict_folder = get_latest_predict_folder()
            if not latest_predict_folder:
                return jsonify({'error': 'Prediction failed - no output folder found'}), 500
            
            # Get the predicted image path
            pred_path = os.path.join(latest_predict_folder, filename)
            logger.debug(f"Looking for prediction at: {pred_path}")
            
            # List contents of the prediction folder for debugging
            try:
                folder_contents = os.listdir(latest_predict_folder)
                logger.debug(f"Contents of prediction folder: {folder_contents}")
            except Exception as e:
                logger.error(f"Error listing folder contents: {str(e)}")
            
            # Move the predicted image to static folder for display
            output_filename = f'pred_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            if os.path.exists(pred_path):
                logger.debug(f"Found prediction file at: {pred_path}")
                # Use shutil.copy2 instead of os.rename to avoid permission issues
                shutil.copy2(pred_path, output_path)
                logger.debug(f"Copied prediction to: {output_path}")
                
                # Clean up the original prediction file
                try:
                    os.remove(pred_path)
                    logger.debug("Cleaned up original prediction file")
                except Exception as e:
                    logger.error(f"Error cleaning up prediction file: {str(e)}")
                
                return jsonify({
                    'success': True,
                    'original_image': f'/static/uploads/{filename}',
                    'predicted_image': f'/static/uploads/{output_filename}'
                })
            else:
                logger.error(f"Prediction file not found at: {pred_path}")
                return jsonify({'error': 'Prediction file not found'}), 500
                
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            # Clean up any temporary files
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
