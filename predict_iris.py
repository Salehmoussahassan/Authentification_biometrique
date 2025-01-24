import numpy as np
import cv2
import os
from keras.models import load_model
import pandas as pd

def predict_iris(image_path):
    model = load_model('static/model/cnn_iris_model.h5', compile=False)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur: Impossible de charger l'image à partir de {image_path}")
        return 0
    
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    probability = np.max(prediction)
    predicted_class = np.argmax(prediction)
    
    print(f"Probabilité prédite : {probability:.2f}, Classe prédite : {predicted_class}")

    if probability < 0.90:
        print(f"Correspondance non trouvée : probabilité {probability:.2f} < 90%")
        return 0
    
    df = pd.read_csv('dataset.csv')
    
    if predicted_class in df['label'].values:
        print(f"Correspondance trouvée avec une probabilité de {probability:.2f}")
        return 1
    else:
        print("Pas de correspondance dans le dataset.")
        return 0
