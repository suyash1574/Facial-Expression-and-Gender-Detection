from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
import json
from datetime import datetime

app = Flask(__name__)

# Hardcoded model paths
EMOTION_MODEL_PATH = "models/emotion_model.h5"
GENDER_MODEL_PATH = "models/gender_model.h5"
AGE_MODEL_PATH = "models/age_model.h5"

# Create directories if they don't exist
SNAPSHOT_DIR = "snapshots"
HISTORY_DIR = "history"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# Load models
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
gender_labels = ['Male', 'Female']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_emotion = "Detecting..."
current_gender = "Detecting..."
current_age = "Detecting..."
current_confidence = "Detecting..."

def gen_frames():
    global current_emotion, current_gender, current_age, current_confidence
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=(0, -1))
            emotion_pred = emotion_model.predict(roi_gray, verbose=0)
            current_emotion = emotion_labels[np.argmax(emotion_pred)]
            current_confidence = (np.max(emotion_pred) * 100).round(2)

            roi_color = frame[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (48, 48))
            roi_color = roi_color.astype('float32') / 255.0
            roi_color_expanded = np.expand_dims(roi_color, axis=0)
            
            gender_pred = gender_model.predict(roi_color_expanded, verbose=0)
            current_gender = gender_labels[np.argmax(gender_pred)]

            age_pred = age_model.predict(roi_color_expanded, verbose=0)
            current_age = f"{int(age_pred[0][0])} years"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\n')

@app.route('/')
def index():
    return render_template('index.html', 
                          emotion=current_emotion, 
                          gender=current_gender, 
                          age=current_age, 
                          confidence=f"{current_confidence}%")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    global current_emotion, current_gender, current_age, current_confidence
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    if success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        
        history_data = {
            "timestamp": timestamp,
            "image_path": filename,
            "gender": current_gender,
            "age": current_age,
            "emotion": current_emotion,
            "confidence": f"{current_confidence}%"
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

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)