
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time

IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 40
MODEL_SAVE_PATH = 'saved_models/emotion_model.keras'
TRAIN_DATA_DIR = 'data/emotion/train'
TEST_DATA_DIR = 'data/emotion/test'
TASK_NAME = "Emotion"

try:
    CLASS_NAMES = sorted(os.listdir(TRAIN_DATA_DIR))
    if not CLASS_NAMES or not os.path.isdir(os.path.join(TRAIN_DATA_DIR, CLASS_NAMES[0])):
         raise FileNotFoundError(f"No valid class subdirectories found in {TRAIN_DATA_DIR}")
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Found {NUM_CLASSES} {TASK_NAME.lower()} classes: {CLASS_NAMES}")
except FileNotFoundError as e:
    print(f"ERROR: Data directory issue for {TASK_NAME}: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

def load_classification_data(train_dir, test_dir, img_height, img_width, batch_size, class_names, channels):
    img_color_mode = 'rgb' if channels == 3 else 'grayscale'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(img_height, img_width), batch_size=batch_size,
        class_mode='categorical', subset='training', color_mode=img_color_mode, classes=class_names)

    print("Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(img_height, img_width), batch_size=batch_size,
        class_mode='categorical', subset='validation', color_mode=img_color_mode, classes=class_names)

    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(img_height, img_width), batch_size=1,
        class_mode='categorical', shuffle=False, color_mode=img_color_mode, classes=class_names)

    if train_generator.samples == 0 or validation_generator.samples == 0 or test_generator.samples == 0:
         print("ERROR: One or more data generators are empty. Check dataset paths and contents.")
         exit()

    return train_generator, validation_generator, test_generator

def build_classifier_model(input_shape, num_classes):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    print(f"--- Training {TASK_NAME} Model ---")
    try:
        train_gen, val_gen, test_gen = load_classification_data(
            TRAIN_DATA_DIR, TEST_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, CLASS_NAMES, CHANNELS
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit()

    input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    model = build_classifier_model(input_shape, NUM_CLASSES)
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    print(f"Starting {TASK_NAME} training...")
    start_time = time.time()
    history = model.fit(
        train_gen, steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
        validation_data=val_gen, validation_steps=max(1, val_gen.samples // BATCH_SIZE),
        epochs=EPOCHS, callbacks=[checkpoint, early_stopping]
    )
    training_time = time.time() - start_time
    print(f"{TASK_NAME} training finished in {training_time:.2f} seconds.")

    print(f"\nEvaluating {TASK_NAME} model...")
    try:
        best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"Could not load best model, using last epoch model: {e}")
        best_model = model

    test_loss, test_acc = best_model.evaluate(test_gen, steps=test_gen.samples)
    print(f'\n{TASK_NAME} Test Accuracy: {test_acc:.4f}')
    print(f'{TASK_NAME} Test Loss: {test_loss:.4f}')

    if test_acc < 0.70:
        print(f"WARNING: {TASK_NAME} model accuracy is below the 70% minimum requirement!")
    else:
        print(f"{TASK_NAME} model accuracy meets the minimum 70% requirement.")

    print("\nGenerating Classification Report and Confusion Matrix...")
    test_gen.reset()
    y_pred = best_model.predict(test_gen, steps=test_gen.samples)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred_classes)
    print(f'\n{TASK_NAME} Classification Report:')
    print(report)
    print(f'\n{TASK_NAME} Confusion Matrix:')
    print(matrix)

    print(f"\n{TASK_NAME} model training complete. Best model potentially saved to {MODEL_SAVE_PATH}")
