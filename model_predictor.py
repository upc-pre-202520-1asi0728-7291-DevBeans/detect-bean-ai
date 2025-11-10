"""
Módulo para carga y uso de modelos de machine learning.
Intenta cargar el modelo CNN real. Si falla, reporta el error.
"""

import numpy as np
from tensorflow import keras

from config import CNN_COLOR_CLASSES, MODEL_PATHS, CNN_CLASS_TO_SCORE_MAP


class ModelPredictor:
    """Clase para manejar predicciones con modelos de ML."""

    def __init__(self):
        """Inicializa el predictor de modelos."""
        self.defect_categories = CNN_COLOR_CLASSES
        self.model_paths = MODEL_PATHS
        self.defect_model = None  # Se inicializa como None
        self.quality_model = None
        self._load_models()

    def _load_models(self):
        """
        Carga el modelo CNN real. Si falla, self.cnn_model permanecerá None.
        """
        try:
            self.defect_model = keras.models.load_model(self.model_paths['defect_detector'])
            print(f"Modelo CNN real cargado exitosamente desde: {self.model_paths['defect_detector']}")
        except Exception as e:
            print(f"ERROR CRÍTICO AL CARGAR MODELO CNN: {e}")
            print(f"El modelo en '{self.model_paths['defect_detector']}' no se pudo cargar.")
            print("El sistema no podrá realizar predicciones de color/calidad.")
            self.defect_model = None

    def predict_color_percentages(self, processed_image):
        """
        Predice los porcentajes de color (Dark, Green, Light, Medium) del grano
        usando el modelo CNN.

        Args:
            processed_image (numpy.ndarray): Imagen procesada del grano (224x224x3)

        Returns:
            dict: Porcentajes para cada clase de color, o None si el modelo no está cargado.
        """
        if self.defect_model is not None:
            # 1. Preparación de la imagen para la CNN
            normalized_image = processed_image / 255.0
            input_tensor = np.expand_dims(normalized_image, axis=0)
            # 2. PREDICCIÓN REAL con el modelo de 4 salidas
            raw_predictions = self.defect_model.predict(input_tensor, verbose=0)[0]
            predictions = {}
            # Mapear la salida del modelo a las clases de color
            for i, color_class in enumerate(CNN_COLOR_CLASSES):
                predictions[color_class] = round(raw_predictions[i].item(), 3)
            # Asegurar que el total sea 100% para la clasificación de rangos
            total_prob = sum(predictions.values())
            if total_prob > 0:
                 predictions = {k: (v / total_prob) * 100 for k, v in predictions.items()}

            return predictions
        # Si el modelo no está cargado, retornar None
        else:
            return None

    @staticmethod
    def get_base_score_from_color(color_percentages):
        """
        Obtiene la puntuación de calidad base encontrando la clase de color
        con el porcentaje (confianza) más alto.

        Args:
            color_percentages (dict): Confianza de la CNN para cada clase.

        Returns:
            tuple (str, float): La clase ganadora y su puntuación base.
        """
        if not color_percentages:
            return 'C', 0.0

        # Encontrar la clase con el valor (confianza) más alto
        # ej: {"Green": 100.0, "Dark": 0.0, ...} -> "Green"
        winning_class = max(color_percentages, key=color_percentages.get)

        # Mapear la clase ganadora a su puntuación de calidad
        # ej: "Green" -> 0.40
        base_score = CNN_CLASS_TO_SCORE_MAP.get(winning_class, 0.0)

        return winning_class, base_score