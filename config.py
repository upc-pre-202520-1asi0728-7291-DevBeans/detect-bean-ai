"""
Configuración global del sistema de clasificación de granos de café.
Define parámetros, rutas y constantes utilizadas en todo el sistema.
"""

# Parámetros de procesamiento de imágenes
IMAGE_SIZE = (224, 224)  # Tamaño estándar para redimensionar imágenes
CONTRAST_FACTOR = 1.5    # Factor para mejorar contraste
BRIGHTNESS_DELTA = 10    # Ajuste de brillo

# Umbrales para clasificación de calidad
QUALITY_THRESHOLDS = {
    'exportacion': 0.8,      # Mínimo 80% de confianza para exportación
    'comercial_local': 0.6,  # Mínimo 60% para comercial local
    'descarte': 0.0         # Por debajo de 60% es descarte
}

# Categorías de clasificación
BEAN_CATEGORIES = [
    'grano_sano',
    'grano_manchado', 
    'grano_quebrado',
    'grano_insecto',
    'grano_moho'
]

# Rutas de modelos (ajustar según sea necesario)
MODEL_PATHS = {
    'defect_detector': 'models/defect_detector.h5',
    'quality_classifier': 'models/quality_classifier.pkl'
}