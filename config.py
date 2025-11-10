"""
Configuración global del sistema de clasificación de granos de café.
Define parámetros, rutas y constantes utilizadas en todo el sistema.
"""

# Parámetros de procesamiento de imágenes
IMAGE_SIZE = (224, 224)  # Tamaño estándar para redimensionar imágenes
CONTRAST_FACTOR = 1.5    # Factor para mejorar contraste
BRIGHTNESS_DELTA = 10    # Ajuste de brillo

QUALITY_THRESHOLDS = {
    'Premium': 0.9,
    'Specialty': 0.8,
    'A': 0.7,
    'B': 0.6,
    'C': 0.0 # 'C' es cualquier cosa por debajo de 'B'
}

# Categorías de CLASIFICACIÓN FINAL
BEAN_CATEGORIES = [
    'Premium',
    'Specialty',
    'A', # Alta calidad comercial
    'B', # Calidad media
    'C'  # Baja calidad
]

# Rutas de modelos (ajustar según sea necesario)
MODEL_PATHS = {
    'defect_detector': 'models/defect_detector.h5',
    'quality_classifier': 'models/quality_classifier.pkl'
}

# Estos son los rangos de una tabla de porcentajes proporcionada para la clasificación final
QUALITY_COLOR_RANGES = {
    'Premium': {'Green': (0, 2), 'Light': (60, 80), 'Medium': (20, 35), 'Dark': (0, 5)},
    'Specialty': {'Green': (0, 5), 'Light': (40, 60), 'Medium': (35, 50), 'Dark': (5, 10)},
    'A': {'Green': (0, 10), 'Light': (25, 40), 'Medium': (40, 60), 'Dark': (10, 20)},
    'B': {'Green': (5, 15), 'Light': (10, 25), 'Medium': (35, 50), 'Dark': (25, 40)},
    'C': {'Green': (10, 30), 'Light': (0, 10), 'Medium': (20, 40), 'Dark': (40, 70)},
}

# Mapeo de categorías de Color/Tueste (salida del CNN)
CNN_COLOR_CLASSES = ['Dark', 'Green', 'Light', 'Medium']