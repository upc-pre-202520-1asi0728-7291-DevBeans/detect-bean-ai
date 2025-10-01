"""
Módulo para clasificación de calidad de granos de café.
Define las reglas de negocio para categorizar los granos.
"""

from config import QUALITY_THRESHOLDS, BEAN_CATEGORIES

class QualityClassifier:
    """Clase para clasificar la calidad de los granos de café."""
    
    def __init__(self):
        """Inicializa el clasificador de calidad."""
        self.quality_thresholds = QUALITY_THRESHOLDS
        self.defect_categories = BEAN_CATEGORIES
    
    def classify_bean_quality(self, defect_predictions, features):
        """
        Clasifica la calidad del grano basado en defectos y características.
        
        Args:
            defect_predictions (dict): Predicciones de tipos de defectos
            features (dict): Características extraídas del grano
            
        Returns:
            dict: Clasificación de calidad y puntuaciones
        """
        # Calcular puntuación base basada en defectos
        defect_score = self._calculate_defect_score(defect_predictions)
        
        # Evaluar características de forma y color
        shape_score = self._evaluate_shape_quality(features)
        color_score = self._evaluate_color_quality(features)
        
        # Puntuación final ponderada
        final_score = (
            defect_score * 0.6 +    # Defectos tienen mayor peso
            shape_score * 0.2 +     # Forma
            color_score * 0.2       # Color
        )
        
        # Determinar categoría de calidad
        quality_category = self._determine_quality_category(final_score)
        
        return {
            'quality_category': quality_category,
            'final_score': final_score,
            'defect_score': defect_score,
            'shape_score': shape_score,
            'color_score': color_score,
            'defect_details': defect_predictions
        }
    
    def _calculate_defect_score(self, defect_predictions):
        """
        Calcula puntuación basada en defectos detectados.
        
        Args:
            defect_predictions (dict): Predicciones de defectos
            
        Returns:
            float: Puntuación de defectos (0-1, donde 1 es sin defectos)
        """
        base_score = 1.0
        
        # Penalizaciones por diferentes tipos de defectos
        penalties = {
            'grano_manchado': 0.3,
            'grano_quebrado': 0.5, 
            'grano_insecto': 0.7,
            'grano_moho': 0.8
        }
        
        for defect_type, confidence in defect_predictions.items():
            if defect_type in penalties and confidence > 0.5:
                base_score -= penalties[defect_type] * confidence
        
        return max(0.0, base_score)  # Asegurar que no sea negativo
    
    def _evaluate_shape_quality(self, features):
        """
        Evalúa la calidad basada en características de forma.
        
        Args:
            features (dict): Características extraídas
            
        Returns:
            float: Puntuación de forma (0-1)
        """
        shape_score = 1.0
        
        # Evaluar circularidad (granos muy irregulares son de menor calidad)
        circularity = features.get('circularity', 0.5)
        if circularity < 0.7:
            shape_score -= 0.3
        elif circularity < 0.5:
            shape_score -= 0.6
        
        # Evaluar área (tamaño consistente)
        area = features.get('area', 0)
        if area < 100:  # Grano muy pequeño
            shape_score -= 0.2
        
        return max(0.0, shape_score)
    
    def _evaluate_color_quality(self, features):
        """
        Evalúa la calidad basada en características de color.
        
        Args:
            features (dict): Características extraídas
            
        Returns:
            float: Puntuación de color (0-1)
        """
        color_score = 1.0
        
        # Granos muy oscuros o muy claros pueden indicar problemas
        value_mean = features.get('value_mean', 128)
        if value_mean < 50:  # Muy oscuro
            color_score -= 0.4
        elif value_mean > 200:  # Muy claro
            color_score -= 0.2
        
        # Saturación muy baja puede indicar granos viejos
        saturation = features.get('saturation_mean', 0)
        if saturation < 30:
            color_score -= 0.3
        
        return max(0.0, color_score)
    
    def _determine_quality_category(self, final_score):
        """
        Determina la categoría de calidad basada en la puntuación final.
        
        Args:
            final_score (float): Puntuación final del grano (0-1)
            
        Returns:
            str: Categoría de calidad
        """
        if final_score >= self.quality_thresholds['exportacion']:
            return 'exportacion'
        elif final_score >= self.quality_thresholds['comercial_local']:
            return 'comercial_local'
        else:
            return 'descarte'
    
    def generate_quality_report(self, beans_classification):
        """
        Genera reporte general de calidad del lote.
        
        Args:
            beans_classification (list): Clasificación de todos los granos
            
        Returns:
            dict: Reporte resumido de calidad del lote
        """
        total_beans = len(beans_classification)
        
        # Contar granos por categoría
        category_count = {
            'exportacion': 0,
            'comercial_local': 0,
            'descarte': 0
        }
        
        for bean in beans_classification:
            category = bean['quality_category']
            category_count[category] += 1
        
        # Calcular porcentajes
        category_percentages = {
            category: (count / total_beans) * 100 
            for category, count in category_count.items()
        }
        
        # Calcular puntuación promedio del lote
        average_score = sum(bean['final_score'] for bean in beans_classification) / total_beans
        
        return {
            'total_beans_analyzed': total_beans,
            'category_distribution': category_count,
            'category_percentages': category_percentages,
            'average_quality_score': average_score,
            'lot_quality': self._determine_quality_category(average_score)
        }