"""
Módulo para carga y uso de modelos de machine learning.
Intenta cargar el modelo CNN real. Si falla, reporta el error.
"""

import numpy as np
from tensorflow import keras

from config import CNN_COLOR_CLASSES, MODEL_PATHS, QUALITY_COLOR_RANGES


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
        Carga los modelos de machine learning.
        Intenta cargar el modelo real .h5; si falla, self.defect_model permanecerá None.
        """
        try:
            # --- MODIFICACIÓN ---
            # Intentar cargar el modelo CNN real (4 clases)
            self.defect_model = keras.models.load_model(self.model_paths['defect_detector'])
            print(f"Modelo CNN real cargado exitosamente desde: {self.model_paths['defect_detector']}")

            # (El clasificador de calidad (pkl) no se usa en este flujo)
            # self.quality_model = joblib.load(self.model_paths['quality_classifier'])

        except Exception as e:
            # --- MODIFICACIÓN ---
            # Si falla la carga, imprime el error y self.defect_model se queda como None.
            print(f"ERROR CRÍTICO AL CARGAR MODELO CNN: {e}")
            print(f"El modelo en '{self.model_paths['defect_detector']}' no se pudo cargar.")
            print("El sistema no podrá realizar predicciones de color/calidad.")
            self.defect_model = None

    def predict_color_percentages(self, processed_image):
        """
        Predice los porcentajes de color (Dark, Green, Light, Medium) del grano
        usando el modelo CNN.

        Args:
            processed_image (numpy.ndarray): Imagen preprocesada del grano (224x224x3)

        Returns:
            dict: Porcentajes para cada clase de color, o None si el modelo no está cargado.
        """
        # --- MODIFICACIÓN ---
        # Comprobar si el modelo se cargó correctamente (no es None)
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

        # --- MODIFICACIÓN ---
        # Si el modelo es None (falló la carga), no hay simulación.
        # Devuelve None para que el bucle principal maneje este error.
        else:
            return None

    @staticmethod
    def classify_by_color_ranges(color_percentages):
        """
        Clasifica la categoría de calidad (Specialty, A, B, C) usando los
        rangos de color predefinidos en config.py.
        (Esta función no cambia)
        """
        for quality_category, ranges in QUALITY_COLOR_RANGES.items():
            is_match = True
            for color_class, (min_val, max_val) in ranges.items():
                predicted_val = color_percentages.get(color_class, 0)

                if not (min_val <= predicted_val <= max_val):
                    is_match = False
                    break
            if is_match:
                return quality_category
        return 'C'