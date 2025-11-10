"""
Programa principal para clasificación automática de granos de café.
Analiza imágenes de granos y clasifica su calidad en categorías.
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime

from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from model_predictor import ModelPredictor
from quality_classifier import QualityClassifier
# config se importa implícitamente a través de los otros módulos

class CoffeeBeanClassifier:
    """Clase principal del sistema de clasificación de granos de café."""

    def __init__(self):
        """Inicializa todos los componentes del sistema."""
        self.image_processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor()
        self.model_predictor = ModelPredictor()
        self.quality_classifier = QualityClassifier()

        print("Sistema de clasificación de granos de café inicializado")

    def analyze_single_image(self, image_path):
        """
        Analiza una sola imagen de granos de café.

        Args:
            image_path (str): Ruta a la imagen a analizar

        Returns:
            dict: Resultados del análisis completo
        """
        print(f"Analizando imagen: {image_path}")

        original_image = self.image_processor.load_image(image_path)
        if original_image is None:
            return {"error": "No se pudo cargar la imagen"}

        print(f"Dimensiones de la imagen original: {original_image.shape}")

        enhanced_image = self.image_processor.enhance_image_quality(original_image)

        beans_data = self.image_processor.segment_beans(enhanced_image)
        print(f"Se encontraron {len(beans_data)} granos en la imagen")

        bean_analysis_results = []

        for i, bean_data in enumerate(beans_data):
            print(f"Analizando grano {i+1}/{len(beans_data)}")

            bean_image = bean_data['image']
            contour = bean_data['contour']

            try:
                features = self.feature_extractor.extract_all_features(bean_image, contour)

                if len(bean_image.shape) == 3:
                    model_input_image = bean_image
                else:
                    model_input_image = cv2.cvtColor(bean_image, cv2.COLOR_GRAY2BGR)

                model_input = cv2.resize(model_input_image, (224, 224))

                # --- INICIO DE MODIFICACIÓN ---

                # 1. Predecir porcentajes de color (Dark, Green, Light, Medium)
                color_percentages = self.model_predictor.predict_color_percentages(model_input)

                # 2. Comprobar si el modelo CNN falló
                if color_percentages is None:
                    print(f"Error en grano {i+1}: El modelo CNN no está cargado. No se puede predecir.")
                    # Registrar un análisis fallido para este grano
                    bean_analysis = {
                        'bean_id': i + 1,
                        'error': 'Modelo CNN no disponible. Predicción fallida.',
                        'quality_assessment': {'quality_category': 'C', 'final_score': 0.0} # Asignar la peor calidad
                    }
                    bean_analysis_results.append(bean_analysis)
                    continue # Saltar al siguiente grano

                # 3. Clasificar la categoría de calidad (Specialty, A, B, C) usando los rangos
                quality_category = self.model_predictor.classify_by_color_ranges(color_percentages)

                # 4. Clasificar calidad final (combinando categoría de color y forma)
                quality_result = self.quality_classifier.classify_bean_quality(
                    quality_category, features
                )

                # --- FIN DE MODIFICACIÓN ---

                bean_analysis = {
                    'bean_id': i + 1,
                    'features': features,
                    'color_percentages': color_percentages,
                    'quality_assessment': quality_result,
                    'bounding_box': bean_data['bbox'],
                    'area': bean_data['area']
                }

                bean_analysis_results.append(bean_analysis)

            except Exception as e:
                print(f"Error analizando grano {i+1}: {e}")
                continue

        # 4. Generar reporte general del lote
        if bean_analysis_results:
            # Filtrar solo los análisis exitosos para el reporte
            successful_assessments = [
                result['quality_assessment'] for result in bean_analysis_results
                if 'error' not in result
            ]

            if successful_assessments:
                 batch_report = self.quality_classifier.generate_quality_report(successful_assessments)
            else:
                 batch_report = {"error": "Ningún grano pudo ser analizado exitosamente (Modelo CNN podría estar fallando)"}
        else:
            batch_report = {"error": "No se pudieron analizar granos"}

        # 5. Compilar resultados finales
        final_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'image_file': image_path,
            'image_dimensions': original_image.shape,
            'total_beans_detected': len(beans_data),
            'beans_analyzed': len(bean_analysis_results),
            'individual_analysis': bean_analysis_results,
            'batch_quality_report': batch_report
        }

        return final_results

    @staticmethod
    def save_results(results, output_path=None):
        """
        Guarda los resultados del análisis en un archivo JSON.
        (Esta función no cambia)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"coffee_analysis_{timestamp}.json"

        try:
            def convert_for_json(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            results_serializable = convert_for_json(results)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)

            print(f"Resultados guardados en: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error guardando resultados: {e}")
            return None

    def print_summary(self, results):
        """
        Imprime un resumen legible de los resultados.
        (Esta función no cambia)
        """
        if 'error' in results:
            print(f"Error en el análisis: {results['error']}")
            return

        batch_report = results.get('batch_quality_report', {})

        print("\n" + "="*50)
        print("RESUMEN DE ANÁLISIS DE GRANOS DE CAFÉ")
        print("="*50)
        print(f"Imagen analizada: {results.get('image_file', 'N/A')}")
        print(f"Dimensiones de la imagen: {results.get('image_dimensions', 'N/A')}")
        print(f"Total de granos detectados: {results.get('total_beans_detected', 0)}")
        print(f"Granos analizados: {results.get('beans_analyzed', 0)}")
        print(f"Fecha del análisis: {results.get('analysis_timestamp', 'N/A')}")

        if batch_report and 'error' not in batch_report:
            print("\n--- CALIDAD DEL LOTE ---")
            print(f"Calificación general: {batch_report.get('lot_quality', 'N/A').upper()}")
            print(f"Puntuación promedio: {batch_report.get('average_quality_score', 0):.3f}")

            distribution = batch_report.get('category_distribution', {})
            percentages = batch_report.get('category_percentages', {})

            print("\nDistribución de calidades:")
            categories_to_print = self.quality_classifier.defect_categories

            for category in categories_to_print:
                count = distribution.get(category, 0)
                percentage = percentages.get(category, 0)
                print(f"  - {category.upper()}: {count} granos ({percentage:.1f}%)")
        else:
            print("\nNo se pudo generar el reporte de calidad del lote (Modelo CNN podría estar fallando)")

        print("="*50)

def main():
    """Función principal del programa."""
    classifier = CoffeeBeanClassifier()

    image_path = "./imagenes_para_analizar/grano1.jpg"

    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen '{image_path}'")
        print("Por favor, coloca una imagen de granos de café en el directorio correcto")
        return

    print("Iniciando análisis de imagen...")
    results = classifier.analyze_single_image(image_path)

    classifier.print_summary(results)

    output_file = classifier.save_results(results)
    
    if output_file:
        print(f"\nResultados detallados guardados en: {output_file}")

if __name__ == "__main__":
    main()