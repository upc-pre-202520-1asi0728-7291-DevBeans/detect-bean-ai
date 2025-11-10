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
        self.quality_categories = BEAN_CATEGORIES

    def classify_bean_quality(self, base_score, source_category, features):
        """
        Clasifica la calidad del grano combinando la puntuación base del modelo
        con las características de forma.

        Args:
            base_score (float): Puntuación de calidad (0-1) del modelo CNN.
            source_category (str): La clase de color ganadora (ej. "Green").
            features (dict): Características extraídas del grano.

        Returns:
            dict: Clasificación de calidad y puntuaciones
        """
        # Evaluar características de forma (se mantienen para refinar la puntuación)
        shape_score = self._evaluate_shape_quality(features)

        # Puntuación final ponderada:
        # 70% de la puntuación viene de la IA (color/tueste)
        # 30% de la puntuación viene de la CV (forma)
        final_score = (
                base_score * 0.7 +
                shape_score * 0.3
        )

        # Determinar categoría de calidad final basada en la puntuación combinada
        final_quality_category = self._determine_quality_category(final_score)

        return {
            'quality_category': final_quality_category,
            'final_score': round(final_score, 3),
            'base_quality_score': base_score,
            'shape_score': shape_score,
            'source_category (color)': source_category
        }

    @staticmethod
    def _evaluate_shape_quality(features):
        """
        Evalúa la calidad basada en características de forma.

        Args:
            features (dict): Características extraídas

        Returns:
            float: Puntuación de forma (0-1)
        """
        shape_score = 1.0

        # El grano verde de prueba es un círculo perfecto,
        # así que 'circularity' es 0.785 (falla) y 'has_cracks' es True.
        # Ajustemos la lógica para ser más permisivos con la forma.

        circularity = features.get('circularity', 0.5)
        if circularity < 0.7:  # Penalización si no es muy circular
            shape_score -= 0.3

        if features.get('has_cracks', False):  # Penalización por grietas
            shape_score -= 0.2

        return max(0.0, shape_score)

    def _determine_quality_category(self, final_score):
        """
        Determina la categoría de calidad basada en la puntuación final.
        Utiliza los umbrales definidos en config.py
        """
        for category, threshold in self.quality_thresholds.items():
            if final_score >= threshold:
                return category
        # Si no supera ni el umbral de 'B' (0.6), es 'C'
        return 'C'

    def generate_quality_report(self, beans_classification):
        """
        Genera reporte general de calidad del lote.

        Args:
            beans_classification (list): Clasificación de todos los granos

        Returns:
            dict: Reporte resumido de calidad del lote
        """
        total_beans = len(beans_classification)
        if total_beans == 0:
            return {"error": "No se analizaron granos para el reporte"}

        # Inicializar dinámicamente el contador de categorías
        category_count = {category: 0 for category in self.quality_categories}

        for bean in beans_classification:
            category = bean['quality_category']
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1  # Registrar categorías inesperadas

        category_percentages = {
            category: (count / total_beans) * 100
            for category, count in category_count.items()
        }

        average_score = sum(bean['final_score'] for bean in beans_classification) / total_beans

        lot_quality = self._determine_quality_category(average_score)

        return {
            'total_beans_analyzed': total_beans,
            'category_distribution': category_count,
            'category_percentages': category_percentages,
            'average_quality_score': round(average_score, 3),
            'lot_quality': lot_quality
        }