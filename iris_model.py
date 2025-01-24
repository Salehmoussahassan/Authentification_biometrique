import os
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Multi-class classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_path):
    # Chargement des images avec les labels sous forme de one-hot encoding
    dataset = image_dataset_from_directory(
        dataset_path,
        image_size=(128, 128),
        batch_size=32,
        label_mode='categorical'  # Pour la classification multi-classes
    )

    num_classes = len(dataset.class_names)
    model = create_model((128, 128, 3), num_classes)
    model.fit(dataset, epochs=10)

    # Sauvegarde du modèle
    model.save('static/model/cnn_iris_model.h5')

    # Enregistrement des labels dans dataset.csv
    image_paths = []
    labels = []

    for image_batch, label_batch in dataset:
        for i in range(len(image_batch)):
            # Conversion du label en classe correspondante
            label_index = np.argmax(label_batch[i])
            labels.append(label_index)

    # Sauvegarder les labels et chemins d'images
    df = pd.DataFrame({'label': labels})
    df.to_csv('dataset.csv', index=False)

if __name__ == "__main__":
    train_model('dataset/CASIA_1/')