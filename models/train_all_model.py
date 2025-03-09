import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Hardcoded paths
PROCESSED_DIR = "data/processed/"

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

# Hardcoded model paths
EMOTION_MODEL_PATH = "models/emotion_model.h5"
GENDER_MODEL_PATH = "models/gender_model.h5"
AGE_MODEL_PATH = "models/age_model.h5"

# Hardcoded model parameters
EMOTION_INPUT_SHAPE = (48, 48, 1)
EMOTION_NUM_CLASSES = 7
EMOTION_EPOCHS = 10
EMOTION_BATCH_SIZE = 64

GENDER_INPUT_SHAPE = (48, 48, 3)
GENDER_NUM_CLASSES = 2
GENDER_EPOCHS = 10
GENDER_BATCH_SIZE = 64

AGE_INPUT_SHAPE = (48, 48, 3)
AGE_EPOCHS = 10
AGE_BATCH_SIZE = 64

def load_preprocessed_data():
    X_train_e = np.load(X_TRAIN_EMOTION)
    y_train_e = np.load(Y_TRAIN_EMOTION)
    X_test_e = np.load(X_TEST_EMOTION)
    y_test_e = np.load(Y_TEST_EMOTION)
    
    X_train_ug = np.load(X_TRAIN_GENDER)
    y_train_g = np.load(Y_TRAIN_GENDER)
    X_test_ug = np.load(X_TEST_GENDER)
    y_test_g = np.load(Y_TEST_GENDER)
    
    y_train_a = np.load(Y_TRAIN_AGE)
    y_test_a = np.load(Y_TEST_AGE)
    
    return (X_train_e, y_train_e, X_test_e, y_test_e), (X_train_ug, y_train_g, X_test_ug, y_test_g, y_train_a, y_test_a)

def build_emotion_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=EMOTION_INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(EMOTION_NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gender_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=GENDER_INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(GENDER_NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_age_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=AGE_INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_all_models():
    (X_train_e, y_train_e, X_test_e, y_test_e), (X_train_ug, y_train_g, X_test_ug, y_test_g, y_train_a, y_test_a) = load_preprocessed_data()

    emotion_model = build_emotion_model()
    emotion_model.fit(X_train_e, y_train_e, epochs=EMOTION_EPOCHS,
                      batch_size=EMOTION_BATCH_SIZE, validation_data=(X_test_e, y_test_e))
    emotion_model.save(EMOTION_MODEL_PATH)
    print(f"Emotion model saved to {EMOTION_MODEL_PATH}")

    gender_model = build_gender_model()
    gender_model.fit(X_train_ug, y_train_g, epochs=GENDER_EPOCHS,
                     batch_size=GENDER_BATCH_SIZE, validation_data=(X_test_ug, y_test_g))
    gender_model.save(GENDER_MODEL_PATH)
    print(f"Gender model saved to {GENDER_MODEL_PATH}")

    age_model = build_age_model()
    age_model.fit(X_train_ug, y_train_a, epochs=AGE_EPOCHS,
                  batch_size=AGE_BATCH_SIZE, validation_data=(X_test_ug, y_test_a))
    age_model.save(AGE_MODEL_PATH)
    print(f"Age model saved to {AGE_MODEL_PATH}")

if __name__ == "__main__":
    train_all_models()