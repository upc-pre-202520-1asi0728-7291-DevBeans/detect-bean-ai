"""
Módulo para procesamiento y mejora de imágenes de granos de café.
Incluye funciones para preprocesamiento y normalización de imágenes.
"""

import cv2
import numpy as np
from config import IMAGE_SIZE, CONTRAST_FACTOR, BRIGHTNESS_DELTA

class ImageProcessor:
    """Clase para procesamiento de imágenes de granos de café."""
    
    def __init__(self):
        """Inicializa el procesador de imágenes."""
        self.image_size = IMAGE_SIZE
    
    def load_image(self, image_path):
        """
        Carga una imagen desde la ruta especificada.
        
        Args:
            image_path (str): Ruta a la imagen a cargar
            
        Returns:
            numpy.ndarray: Imagen cargada en formato BGR
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen desde: {image_path}")
            return image
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return None
    
    def preprocess_image(self, image):
        """
        Preprocesa la imagen para mejorar la detección de características.
        
        Args:
            image (numpy.ndarray): Imagen original en formato BGR
            
        Returns:
            numpy.ndarray: Imagen preprocesada y normalizada
        """
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Aplicar filtro Gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Mejorar contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Redimensionar a tamaño estándar
        resized = cv2.resize(enhanced, self.image_size)
        
        # Normalizar valores de píxeles
        normalized = resized / 255.0
        
        return normalized
    
    def enhance_image_quality(self, image):
        """
        Mejora la calidad de la imagen para mejor análisis.
        
        Args:
            image (numpy.ndarray): Imagen original
            
        Returns:
            numpy.ndarray: Imagen mejorada
        """
        # Asegurarse de que la imagen esté en formato BGR
        if len(image.shape) == 2:  # Si es escala de grises
            enhanced = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = image.copy()
        
        # Ajustar brillo y contraste
        enhanced = cv2.convertScaleAbs(enhanced, alpha=CONTRAST_FACTOR, beta=BRIGHTNESS_DELTA)
        
        # Reducir ruido preservando bordes
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def segment_beans(self, image):
        """
        Segmenta granos individuales en la imagen.
        
        Args:
            image (numpy.ndarray): Imagen mejorada
            
        Returns:
            list: Lista de granos segmentados individualmente
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Aplicar umbralización para separar granos del fondo
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para limpiar la imagen
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        beans = []
        for i, contour in enumerate(contours):
            # Filtrar contornos pequeños (ruido)
            area = cv2.contourArea(contour)
            if area > 500:  # Aumentar el área mínima para filtrar mejor
                x, y, w, h = cv2.boundingRect(contour)
                
                # Asegurarse de que las coordenadas estén dentro de los límites
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w > 0 and h > 0:  # Verificar dimensiones válidas
                    if len(image.shape) == 3:
                        bean = image[y:y+h, x:x+w]
                    else:
                        bean = image[y:y+h, x:x+w]
                    
                    beans.append({
                        'image': bean,
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        print(f"Segmentados {len(beans)} granos válidos")
        return beans