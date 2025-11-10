"""
Configuración global del sistema de clasificación de granos de café.
Define parámetros, rutas y constantes utilizadas en todo el sistema.
"""

# Parámetros de procesamiento de imágenes
IMAGE_SIZE = (224, 224)  # Tamaño estándar para redimensionar imágenes
CONTRAST_FACTOR = 1.5    # Factor para mejorar contraste
BRIGHTNESS_DELTA = 10    # Ajuste de brillo

# Umbrales para clasificación de calidad (para la puntuación final)
QUALITY_THRESHOLDS = {
    'Specialty': 0.9,
    'Premium': 0.8,
    'A': 0.7,
    'B': 0.6,
    'C': 0.0 # 'C' es cualquier cosa por debajo de 'B'
}

# Categorías de CLASIFICACIÓN FINAL (del negocio)
BEAN_CATEGORIES = [
    'Specialty',
    'Premium',
    'A', # Alta calidad comercial
    'B', # Calidad media
    'C'  # Baja calidad / industrial
]

# Rutas de modelos
MODEL_PATHS = {
    'defect_detector': 'models/defect_detector.h5',
    'quality_classifier': 'models/quality_classifier.pkl'
}

# Mapeo de categorías de Color/Tueste (salida del CNN)
CNN_COLOR_CLASSES = ['Dark', 'Green', 'Light', 'Medium']

# --- NUEVA LÓGICA DE MAPEO EN BASE LAS VARIABLES DEL REPOSITORIO DE KAGGLE ---
# Mapea la clase ganadora de la CNN a una puntuación de calidad base.
# (Estos valores se basan en una tabla, donde 'Light' es el mejor y 'Green'/'Dark' son los peores)
CNN_CLASS_TO_SCORE_MAP = {
    'Light': 0.95,   # 'Light' (60-80%) es la base de 'Specialty'
    'Medium': 0.85,  # 'Medium' (35-50%) es la base de 'Premium'
    'Dark': 0.40,    # 'Dark' (40-70%) es la base de 'C'
    'Green': 0.40    # 'Green' (10-30%) es la base de 'C'
}