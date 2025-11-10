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
        # 'defect_categories' ahora contiene las categorías de CALIDAD
        self.defect_categories = BEAN_CATEGORIES

    def classify_bean_quality(self, quality_category, features):
        """
        Clasifica la calidad del grano basado en la categoría de color/tueste
        y las características de forma.

        Args:
            quality_category (str): Categoría de calidad determinada por los rangos de color (Premium, Specialty, A, B, C)
            features (dict): Características extraídas del grano

        Returns:
            dict: Clasificación de calidad y puntuaciones
        """
        # Calcular puntuación base basada en la categoría de calidad
        base_score = self._calculate_quality_score(quality_category)

        # Evaluar características de forma (se mantienen para refinar la puntuación)
        shape_score = self._evaluate_shape_quality(features)

        # Puntuación final ponderada: La categoría de color/calidad tiene el mayor peso
        final_score = (
                base_score * 0.7 +  # Categoría de color/calidad
                shape_score * 0.3  # Forma
        )

        # Determinar categoría de calidad final (puede ser refinado por forma)
        final_quality_category = self._determine_quality_category(final_score)

        return {
            'quality_category': final_quality_category,
            'final_score': final_score,
            'base_quality_score': base_score,
            'shape_score': shape_score,
            'source_category': quality_category
        }

    @staticmethod
    def _calculate_quality_score(quality_category):
        """
        Asigna una puntuación numérica a la categoría de calidad determinada.

        Args:
            quality_category (str): Categoría de calidad (Premium, Specialty, A, B, C)

        Returns:
            float: Puntuación de calidad (0-1)
        """
        # Mapeo de categorías a puntajes base (siguiendo los umbrales de exportación)
        if quality_category == 'Premium':
            return 0.95
        elif quality_category == 'Specialty':
            return 0.85
        elif quality_category == 'A':
            return 0.75
        elif quality_category == 'B':
            return 0.60
        elif quality_category == 'C':
            return 0.40
        else:
            return 0.0

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

        circularity = features.get('circularity', 0.5)
        if circularity < 0.7:
            shape_score -= 0.3
        elif circularity < 0.5:
            shape_score -= 0.6

        area = features.get('area', 0)
        if area < 100:  # Grano muy pequeño
            shape_score -= 0.2

        return max(0.0, shape_score)

    @staticmethod
    def _evaluate_color_quality(features):
        """
        Evalúa la calidad basada en características de color. (No se usa actualmente
        en la ponderación principal, pero es útil para análisis de características)
        """
        color_score = 1.0
        value_mean = features.get('value_mean', 128)
        if value_mean < 50:
            color_score -= 0.4
        elif value_mean > 200:
            color_score -= 0.2

        saturation = features.get('saturation_mean', 0)
        if saturation < 30:
            color_score -= 0.3

        return max(0.0, color_score)

    def _determine_quality_category(self, final_score):
        """
        Determina la categoría de calidad basada en la puntuación final.
        Utiliza los umbrales definidos en config.py
        """
        # Iterar desde la categoría más alta a la más baja
        for category, threshold in self.quality_thresholds.items():
            if final_score >= threshold:
                return category
        return 'C' # Categoría por defecto si no supera ni el mínimo de 'B'

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
        # usando las categorías de CALIDAD del negocio (Premium, Specialty, A, B, C)
        category_count = {category: 0 for category in self.defect_categories}

        for bean in beans_classification:
            category = bean['quality_category']
            # Asegurarse de que la categoría exista antes de sumar
            if category in category_count:
                category_count[category] += 1
            else:
                # Si una categoría inesperada aparece, la registra
                category_count[category] = 1

        # Calcular porcentajes
        category_percentages = {
            category: (count / total_beans) * 100
            for category, count in category_count.items()
        }

        # Calcular puntuación promedio del lote
        average_score = sum(bean['final_score'] for bean in beans_classification) / total_beans

        # Determinar la calidad general del lote
        lot_quality = self._determine_quality_category(average_score)

        return {
            'total_beans_analyzed': total_beans,
            'category_distribution': category_count,
            'category_percentages': category_percentages,
            'average_quality_score': average_score,
            'lot_quality': lot_quality
        }