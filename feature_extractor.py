"""
Módulo para extracción de características de granos de café.
Incluye análisis de color, textura, forma y defectos.
"""

import cv2
import numpy as np
from skimage import feature, measure
from config import BEAN_CATEGORIES

class FeatureExtractor:
    """Clase para extracción de características de granos de café."""
    
    def __init__(self):
        """Inicializa el extractor de características."""
        self.categories = BEAN_CATEGORIES
    
    def extract_color_features(self, image):
        """
        Extrae características de color del grano.
        
        Args:
            image (numpy.ndarray): Imagen del grano
            
        Returns:
            dict: Características de color (promedio, desviación, histograma)
        """
        color_features = {}
        
        try:
            # Verificar si la imagen es a color
            if len(image.shape) == 3:
                # Convertir a diferentes espacios de color
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                
                # Características en espacio HSV
                color_features['hue_mean'] = np.mean(hsv[:, :, 0])
                color_features['saturation_mean'] = np.mean(hsv[:, :, 1])
                color_features['value_mean'] = np.mean(hsv[:, :, 2])
                
                # Características en espacio LAB
                color_features['lightness_mean'] = np.mean(lab[:, :, 0])
                color_features['a_channel_mean'] = np.mean(lab[:, :, 1])
                color_features['b_channel_mean'] = np.mean(lab[:, :, 2])
            else:
                # Para imágenes en escala de grises
                color_features['intensity_mean'] = np.mean(image)
                color_features['intensity_std'] = np.std(image)
                
        except Exception as e:
            print(f"Error extrayendo características de color: {e}")
            # Valores por defecto
            color_features.update({
                'hue_mean': 0, 'saturation_mean': 0, 'value_mean': 0,
                'lightness_mean': 0, 'a_channel_mean': 0, 'b_channel_mean': 0
            })
        
        return color_features
    
    def extract_texture_features(self, image):
        """
        Extrae características de textura usando LBP (Local Binary Patterns).
        
        Args:
            image (numpy.ndarray): Imagen en escala de grises
            
        Returns:
            dict: Características de textura
        """
        texture_features = {}
        
        try:
            # Asegurarse de que la imagen esté en escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calcular LBP
            radius = 3
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calcular histograma de LBP
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)  # Normalizar
            
            texture_features = {
                'lbp_mean': np.mean(lbp),
                'lbp_std': np.std(lbp),
                'lbp_histogram': hist
            }
            
        except Exception as e:
            print(f"Error extrayendo características de textura: {e}")
            texture_features = {
                'lbp_mean': 0, 'lbp_std': 0, 'lbp_histogram': []
            }
        
        return texture_features
    
    def extract_shape_features(self, contour):
        """
        Extrae características de forma del grano.
        
        Args:
            contour (numpy.ndarray): Contorno del grano
            
        Returns:
            dict: Características de forma
        """
        shape_features = {}
        
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calcular características de forma
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            # Momentos de Hu (invariantes a transformaciones)
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            shape_features = {
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'hu_moments': hu_moments.tolist()  # Convertir a lista para serialización
            }
            
        except Exception as e:
            print(f"Error extrayendo características de forma: {e}")
            shape_features = {
                'area': 0, 'perimeter': 0, 'circularity': 0, 'hu_moments': [0]*7
            }
        
        return shape_features
    
    def detect_defects(self, image):
        """
        Detecta defectos visibles en el grano.
        
        Args:
            image (numpy.ndarray): Imagen del grano
            
        Returns:
            dict: Información sobre defectos detectados
        """
        defects = {}
        
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detectar bordes para encontrar quiebres
            edges = cv2.Canny(gray, 50, 150)
            
            # Detectar manchas (áreas oscuras)
            _, dark_spots = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Detectar áreas irregulares
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defects = {
                'edge_density': np.sum(edges) / (255 * edges.size) if edges.size > 0 else 0,
                'dark_spots_area': np.sum(dark_spots) / (255 * dark_spots.size) if dark_spots.size > 0 else 0,
                'irregular_contours': len(contours),
                'has_cracks': len(contours) > 5  # Muchos contornos pequeños indican quiebres
            }
            
        except Exception as e:
            print(f"Error detectando defectos: {e}")
            defects = {
                'edge_density': 0, 'dark_spots_area': 0, 
                'irregular_contours': 0, 'has_cracks': False
            }
        
        return defects
    
    def extract_all_features(self, image, contour):
        """
        Extrae todas las características del grano.
        
        Args:
            image (numpy.ndarray): Imagen del grano
            contour (numpy.ndarray): Contorno del grano
            
        Returns:
            dict: Todas las características extraídas
        """
        features = {}
        
        # Extraer características de color
        color_features = self.extract_color_features(image)
        features.update(color_features)
        
        # Extraer características de textura
        texture_features = self.extract_texture_features(image)
        features.update(texture_features)
        
        # Extraer características de forma
        shape_features = self.extract_shape_features(contour)
        features.update(shape_features)
        
        # Detectar defectos
        defect_features = self.detect_defects(image)
        features.update(defect_features)
        
        return features