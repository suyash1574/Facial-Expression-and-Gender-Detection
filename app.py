from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os
import json
from datetime import datetime
import os
import shutil

app = Flask(__name__)

# Hardcoded model paths
EMOTION_MODEL_PATH = "models/emotion_model.h5"
GENDER_MODEL_PATH = "models/gender_model.h5"
AGE_MODEL_PATH = "models/age_model.h5"

# Create directories if they don't exist
SNAPSHOT_DIR = "snapshots"
HISTORY_DIR = "history"
UPLOAD_DIR = "uploads"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load models
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
gender_labels = ['Male', 'Female']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize with "Detecting..." and store last detected values
current_emotion = "Detecting..."
current_gender = "Detecting..."
current_age = "Detecting..."
current_confidence = "Detecting..."
last_detected_emotion = "Detecting..."
last_detected_gender = "Detecting..."
last_detected_age = "Detecting..."
last_detected_confidence = "Detecting..."

# Check if running on Render (e.g., no camera available)
IS_RENDER = os.getenv('IS_RENDER', 'false').lower() == 'true'

def process_image(image_path):
    global current_emotion, current_gender, current_age, current_confidence
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image from {image_path}")
        return False, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=(0, -1))
        emotion_pred = emotion_model.predict(roi_gray, verbose=0)
        emotion = emotion_labels[np.argmax(emotion_pred)]
        confidence = (np.max(emotion_pred) * 100).round(2)

        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (48, 48))
        roi_color = roi_color.astype('float32') / 255.0
        roi_color_expanded = np.expand_dims(roi_color, axis=0)
        
        gender_pred = gender_model.predict(roi_color_expanded, verbose=0)
        gender = gender_labels[np.argmax(gender_pred)]

        age_pred = age_model.predict(roi_color_expanded, verbose=0)
        age = f"{int(age_pred[0][0])} years"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print("Image processed successfully.")
        return True, (emotion, gender, age, confidence)
    else:
        print("No face detected in the image.")
        return False, None

def gen_frames():
    global current_emotion, current_gender, current_age, current_confidence, last_detected_emotion, last_detected_gender, last_detected_age, last_detected_confidence
    if IS_RENDER:
        # On Render, use fallback image and process it
        frame = cv2.imread("fallback.jpg")
        if frame is None:
            print("Fallback image not found. Using blank frame.")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Process fallback image to simulate detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=(0, -1))
                emotion_pred = emotion_model.predict(roi_gray, verbose=0)
                emotion = emotion_labels[np.argmax(emotion_pred)]
                confidence = (np.max(emotion_pred) * 100).round(2)
                roi_color = frame[y:y+h, x:x+w]
                roi_color = cv2.resize(roi_color, (48, 48))
                roi_color = roi_color.astype('float32') / 255.0
                roi_color_expanded = np.expand_dims(roi_color, axis=0)
                gender_pred = gender_model.predict(roi_color_expanded, verbose=0)
                gender = gender_labels[np.argmax(gender_pred)]
                age_pred = age_model.predict(roi_color_expanded, verbose=0)
                age = f"{int(age_pred[0][0])} years"
                current_emotion = emotion
                current_gender = gender
                current_age = age
                current_confidence = confidence
    else:
        # Locally, use camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Camera not available locally. Using fallback image.")
            frame = cv2.imread("fallback.jpg")
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            print("Camera initialized successfully.")
            while True:
                success, frame = camera.read()
                if not success:
                    print("Camera feed failed. Using fallback image.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    if not IS_RENDER:  # Update analysis from camera locally
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (48, 48))
                        roi_gray = roi_gray.astype('float32') / 255.0
                        roi_gray = np.expand_dims(roi_gray, axis=(0, -1))
                        emotion_pred = emotion_model.predict(roi_gray, verbose=0)
                        emotion = emotion_labels[np.argmax(emotion_pred)]
                        confidence = (np.max(emotion_pred) * 100).round(2)
                        roi_color = frame[y:y+h, x:x+w]
                        roi_color = cv2.resize(roi_color, (48, 48))
                        roi_color = roi_color.astype('float32') / 255.0
                        roi_color_expanded = np.expand_dims(roi_color, axis=0)
                        gender_pred = gender_model.predict(roi_color_expanded, verbose=0)
                        gender = gender_labels[np.argmax(gender_pred)]
                        age_pred = age_model.predict(roi_color_expanded, verbose=0)
                        age = f"{int(age_pred[0][0])} years"
                        # Update last detected values
                        last_detected_emotion = emotion
                        last_detected_gender = gender
                        last_detected_age = age
                        last_detected_confidence = confidence
                        # Set current values to last detected
                        current_emotion = last_detected_emotion
                        current_gender = last_detected_gender
                        current_age = last_detected_age
                        current_confidence = last_detected_confidence
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\n')
            camera.release()

    # Continuous frame generation for feed on Render
    while True:
        if IS_RENDER and frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\n')

@app.route('/')
def index():
    camera_message = "Camera feed is not available on this platform. Please use the upload feature." if IS_RENDER else ""
    return render_template('index.html', 
                          emotion=current_emotion, 
                          gender=current_gender, 
                          age=current_age, 
                          confidence=f"{current_confidence}%",
                          camera_message=camera_message)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_page')
def upload_page():
    global current_emotion, current_gender, current_age, current_confidence
    # Initialize with "Detecting..." on page load
    current_emotion = "Detecting..."
    current_gender = "Detecting..."
    current_age = "Detecting..."
    current_confidence = "Detecting..."
    # Check for the latest uploaded image only if it exists
    last_upload = [f for f in os.listdir(UPLOAD_DIR) if f.startswith('upload_')]
    face_not_detected = False
    if last_upload:
        filepath = os.path.join(UPLOAD_DIR, max(last_upload))
        success, result = process_image(filepath)
        if success and result:
            current_emotion, current_gender, current_age, current_confidence = result
        else:
            face_not_detected = True
    return render_template('upload.html',
                          emotion=current_emotion,
                          gender=current_gender,
                          age=current_age,
                          confidence=f"{current_confidence}%",
                          face_not_detected=face_not_detected)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"message": "No selected file"}), 400
    if file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        print(f"Uploaded image saved as {filepath}")

        # Process the uploaded image
        success, result = process_image(filepath)
        if success and result:
            current_emotion, current_gender, current_age, current_confidence = result
            # Save to history
            snapshot_filename = f"snapshot_{timestamp}.jpg"
            snapshot_filepath = os.path.join(SNAPSHOT_DIR, snapshot_filename)
            uploaded_frame = cv2.imread(filepath)
            if uploaded_frame is not None:
                cv2.imwrite(snapshot_filepath, uploaded_frame)
                history_data = {
                    "timestamp": timestamp,
                    "image_path": snapshot_filename,
                    "gender": current_gender,
                    "age": current_age,
                    "emotion": current_emotion,
                    "confidence": f"{current_confidence}%"
                }
                history_file = os.path.join(HISTORY_DIR, f"history_{timestamp}.json")
                with open(history_file, 'w') as f:
                    json.dump(history_data, f)
            print("Upload data saved to history.")
            return redirect(url_for('upload_page'))
        else:
            return redirect(url_for('upload_page', face_not_detected=True))

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    global current_emotion, current_gender, current_age, current_confidence
    # Capture current frame from camera if active
    if not IS_RENDER and last_detected_emotion != "Detecting...":
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            success, frame = camera.read()
            camera.release()
            if success and frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(filepath, frame)
                # Process the frame for accurate data
                success, result = process_image(filepath)
                if success and result:
                    emotion, gender, age, confidence = result
                else:
                    emotion, gender, age, confidence = last_detected_emotion, last_detected_gender, last_detected_age, last_detected_confidence
            else:
                frame = cv2.imread("fallback.jpg")
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(filepath, frame)
                emotion, gender, age, confidence = last_detected_emotion, last_detected_gender, last_detected_age, last_detected_confidence
        else:
            frame = cv2.imread("fallback.jpg")
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(filepath, frame)
            emotion, gender, age, confidence = last_detected_emotion, last_detected_gender, last_detected_age, last_detected_confidence
    else:
        last_upload = [f for f in os.listdir(UPLOAD_DIR) if f.startswith('upload_')]
        if last_upload:
            frame = cv2.imread(os.path.join(UPLOAD_DIR, max(last_upload)))
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(filepath, frame)
            # Process the frame for accurate data
            success, result = process_image(filepath)
            if success and result:
                emotion, gender, age, confidence = result
            else:
                emotion, gender, age, confidence = current_emotion, current_gender, current_age, current_confidence
        else:
            frame = cv2.imread("fallback.jpg")
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(filepath, frame)
            emotion, gender, age, confidence = current_emotion, current_gender, current_age, current_confidence

    if frame is not None:
        history_data = {
            "timestamp": timestamp,
            "image_path": filename,
            "gender": gender,
            "age": age,
            "emotion": emotion,
            "confidence": f"{confidence}%"
        }
        history_file = os.path.join(HISTORY_DIR, f"history_{timestamp}.json")
        with open(history_file, 'w') as f:
            json.dump(history_data, f)
        
        return jsonify({"message": "Snapshot saved", "filename": filename})
    return jsonify({"message": "Failed to save snapshot"}), 500

@app.route('/history')
def history():
    history_files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
    history_data = []
    for file in history_files:
        with open(os.path.join(HISTORY_DIR, file), 'r') as f:
            data = json.load(f)
            history_data.append(data)
    return render_template('history.html', history=history_data)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Delete all files in snapshots directory
        for file in os.listdir(SNAPSHOT_DIR):
            file_path = os.path.join(SNAPSHOT_DIR, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        # Delete all files in history directory
        for file in os.listdir(HISTORY_DIR):
            file_path = os.path.join(HISTORY_DIR, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({"message": "History cleared successfully"})
    except Exception as e:
        return jsonify({"message": f"Failed to clear history: {str(e)}"}), 500

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)