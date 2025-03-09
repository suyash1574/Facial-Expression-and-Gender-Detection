import numpy as np
import os
import cv2
import tensorflow as tf

# Hardcoded paths
EXPRESSION_DIR = "data/raw/expression/"  # FER-2013
UTKFACE_DIR = "data/raw/gender/UTKFace/"  # UTKFace
PROCESSED_DIR = "data/processed/"

# Hardcoded preprocessed file paths
X_TRAIN_EMOTION = os.path.join(PROCESSED_DIR, "X_train_emotion.npy")
Y_TRAIN_EMOTION = os.path.join(PROCESSED_DIR, "y_train_emotion.npy")
X_TEST_EMOTION = os.path.join(PROCESSED_DIR, "X_test_emotion.npy")
Y_TEST_EMOTION = os.path.join(PROCESSED_DIR, "y_test_emotion.npy")

X_TRAIN_GENDER = os.path.join(PROCESSED_DIR, "X_train_gender.npy")
Y_TRAIN_GENDER = os.path.join(PROCESSED_DIR, "y_train_gender.npy")
X_TEST_GENDER = os.path.join(PROCESSED_DIR, "X_test_gender.npy")
Y_TEST_GENDER = os.path.join(PROCESSED_DIR, "y_test_gender.npy")

X_TRAIN_AGE = os.path.join(PROCESSED_DIR, "X_train_age.npy")
Y_TRAIN_AGE = os.path.join(PROCESSED_DIR, "y_train_age.npy")
X_TEST_AGE = os.path.join(PROCESSED_DIR, "X_test_age.npy")
Y_TEST_AGE = os.path.join(PROCESSED_DIR, "y_test_age.npy")

# Hardcoded model parameters
EMOTION_NUM_CLASSES = 7  # For FER-2013 (7 emotions)

def preprocess_expression(data_dir):
    X, y = [], []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for split in ['train', 'test']:
        for i, emotion in enumerate(emotions):
            folder = os.path.join(data_dir, split, emotion)
            for img_file in os.listdir(folder):
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48)) / 255.0
                X.append(img)
                y.append(i)
    X = np.array(X).reshape(-1, 48, 48, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=EMOTION_NUM_CLASSES)
    
    X_train, y_train = X[:28709], y[:28709]
    X_test, y_test = X[28709:], y[28709:]
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(X_TRAIN_EMOTION, X_train)
    np.save(Y_TRAIN_EMOTION, y_train)
    np.save(X_TEST_EMOTION, X_test)
    np.save(Y_TEST_EMOTION, y_test)
    print("Expression (FER-2013) preprocessing completed")

def preprocess_utkface(data_dir):
    X, y_gender, y_age = [], [], []
    for img_file in os.listdir(data_dir):
        if img_file.endswith('.jpg'):
            parts = img_file.split('_')
            if len(parts) >= 3:
                age = min(int(parts[0]), 100)
                gender = int(parts[1])
                img_path = os.path.join(data_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (48, 48)) / 255.0
                    X.append(img)
                    y_gender.append(gender)
                    y_age.append(age)
    X = np.array(X)
    y_gender = tf.keras.utils.to_categorical(y_gender, num_classes=2)  # 2 classes: male, female
    y_age = np.array(y_age)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_gender_train, y_gender_test = y_gender[:split], y_gender[split:]
    y_age_train, y_age_test = y_age[:split], y_age[split:]
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(X_TRAIN_GENDER, X_train)
    np.save(Y_TRAIN_GENDER, y_gender_train)
    np.save(X_TEST_GENDER, X_test)
    np.save(Y_TEST_GENDER, y_gender_test)
    np.save(X_TRAIN_AGE, X_train)
    np.save(Y_TRAIN_AGE, y_age_train)
    np.save(X_TEST_AGE, X_test)
    np.save(Y_TEST_AGE, y_age_test)
    print("UTKFace (gender/age) preprocessing completed")

if __name__ == "__main__":
    preprocess_expression(EXPRESSION_DIR)
    preprocess_utkface(UTKFACE_DIR)