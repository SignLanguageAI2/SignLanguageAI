import numpy as np
import mediapipe as mp
import os
from tensorflow import keras

class SignLanguageDetector:
    def __init__(self):
        # Initialisation de MediaPipe pour la détection des mains
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Labels pour les lettres (alphabet ASL)
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                      'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Modèle (sera chargé ou créé)
        self.model = None
        self.load_or_create_model()

    def load_or_create_model(self):
        """Charge un modèle existant ou en crée un nouveau"""
        model_path = 'sign_language_model.h5'
        if os.path.exists(model_path):
            print("Chargement du modèle existant...")
            self.model = keras.models.load_model(model_path)
        else:
            print("Création d'un nouveau modèle...")
            self.model = self.create_model()
            # Ici vous pourriez ajouter l'entraînement avec un dataset
            print("Modèle créé. Entraînement nécessaire avec un dataset.")

    def create_model(self):
        """Crée un modèle simple"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(63,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

