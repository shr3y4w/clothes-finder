from flask import Flask, render_template, request, redirect, url_for,send_file,jsonify,Response
import os
from werkzeug.utils import secure_filename
from image_similarity import find_similar_images  
from feature_extraction import load_features  
import cv2
import numpy as np
import mediapipe as mp
from overlaytest import overlay_clothing
from datetime import datetime
import hashlib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['CAPTURES_FOLDER'] = os.path.join('static', 'captures')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURES_FOLDER'], exist_ok=True)
image_features = load_features()

# In memory cache for query results
query_cache = {}
CACHE_SIZE_LIMIT = 200 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_cache_key(file_path, color_weight, sift_weight, lbp_weight):
    """Generate a cache key based on file content and weights"""
    # Get file modification time and size for cache key
    stat = os.stat(file_path)
    cache_data = f"{file_path}_{stat.st_mtime}_{stat.st_size}_{color_weight}_{sift_weight}_{lbp_weight}"
    return hashlib.md5(cache_data.encode()).hexdigest()

def cleanup_cache():
    """Remove oldest entries if cache is too large"""
    if len(query_cache) > CACHE_SIZE_LIMIT:
        # Remove oldest entries (simple FIFO)
        oldest_keys = list(query_cache.keys())[:len(query_cache) - CACHE_SIZE_LIMIT]
        for key in oldest_keys:
            del query_cache[key]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/results', methods=['POST'])
def results():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        color_weight = 0.8
        sift_weight = 0.5
        lbp_weight = 0.6
        
        # Check cache first
        cache_key = get_cache_key(file_path, color_weight, sift_weight, lbp_weight)
        if cache_key in query_cache:
            print(f"Cache hit for {filename}")
            similar_images = query_cache[cache_key]
        else:
            print(f"Cache miss for {filename}, computing similarity...")
            similar_images = find_similar_images(file_path, image_features, color_weight, sift_weight, lbp_weight, top_k=50)
            
            # Cache the result
            query_cache[cache_key] = similar_images
            cleanup_cache()
        
        print(f"Query image: {filename}")  
        print(f"Similar images: {similar_images}") 

        return render_template('results.html', query_image=filename, similar_images=similar_images)
    return redirect(url_for('index'))


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
selected_clothing = None

def remove_background(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 245, 255, cv2.THRESH_BINARY_INV) 

    if image.shape[2] == 4:
        b, g, r, _ = cv2.split(image)
    elif image.shape[2] == 3:
        b, g, r = cv2.split(image)
    else:
        print(f"Unexpected number of channels: {image.shape[2]}")
        return None
    rgba_image = cv2.merge((b, g, r, mask))
    return rgba_image

def generate_frames():
    global selected_clothing
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam")
            break
        else:
            try:
                if selected_clothing is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(frame_rgb)
                    
                    if result.pose_landmarks:
                        frame = overlay_clothing(selected_clothing, frame, result.pose_landmarks.landmark)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Error: Could not encode frame")
                    continue
                    
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Error in frame processing: {str(e)}")
                continue
    
    cap.release()

@app.route('/try_on/<image_name>')
def try_on(image_name):
    global selected_clothing
    image_path = os.path.join('static', 'cloth', image_name)
    selected_clothing = remove_background(image_path)
    if selected_clothing is None:
        return "Error: Could not load clothing image", 400
    return render_template('try_on.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)