"""
Programa principal para clasificación automática de granos de café.
Analiza imágenes de granos y clasifica su calidad en categorías.
"""

import os
import cv2
import json
from datetime import datetime

from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from model_predictor import ModelPredictor
from quality_classifier import QualityClassifier
from config import BEAN_CATEGORIES

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
        
        # 1. Cargar y preprocesar imagen
        original_image = self.image_processor.load_image(image_path)
        if original_image is None:
            return {"error": "No se pudo cargar la imagen"}
        
        print(f"Dimensiones de la imagen original: {original_image.shape}")
        
        # Preprocesar para ML (escala de grises)
        processed_image = self.image_processor.preprocess_image(original_image)
        
        # Mejorar para segmentación (color)
        enhanced_image = self.image_processor.enhance_image_quality(original_image)
        
        # 2. Segmentar granos individuales
        beans_data = self.image_processor.segment_beans(enhanced_image)
        print(f"Se encontraron {len(beans_data)} granos en la imagen")
        
        # 3. Analizar cada grano individualmente
        bean_analysis_results = []
        
        for i, bean_data in enumerate(beans_data):
            print(f"Analizando grano {i+1}/{len(beans_data)}")
            
            bean_image = bean_data['image']
            contour = bean_data['contour']
            
            try:
                # Extraer características
                features = self.feature_extractor.extract_all_features(bean_image, contour)
                
                # Predecir defectos - usar una versión procesada del grano individual
                if len(bean_image.shape) == 3:
                    gray_bean = cv2.cvtColor(bean_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_bean = bean_image
                
                # Redimensionar y normalizar para el modelo
                model_input = cv2.resize(gray_bean, (224, 224)) / 255.0
                defect_predictions = self.model_predictor.predict_defects(model_input)
                
                # Clasificar calidad
                quality_result = self.quality_classifier.classify_bean_quality(
                    defect_predictions, features
                )
                
                bean_analysis = {
                    'bean_id': i + 1,
                    'features': features,
                    'defect_predictions': defect_predictions,
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
            quality_assessments = [result['quality_assessment'] for result in bean_analysis_results]
            batch_report = self.quality_classifier.generate_quality_report(quality_assessments)
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
    
    def save_results(self, results, output_path=None):
        """
        Guarda los resultados del análisis en un archivo JSON.
        
        Args:
            results (dict): Resultados del análisis
            output_path (str): Ruta donde guardar los resultados
            
        Returns:
            str: Ruta del archivo guardado
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"coffee_analysis_{timestamp}.json"
        
        try:
            # Convertir numpy arrays a listas para JSON
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
        
        Args:
            results (dict): Resultados del análisis
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
            for category, count in distribution.items():
                percentage = percentages.get(category, 0)
                print(f"  - {category.upper()}: {count} granos ({percentage:.1f}%)")
        else:
            print("\nNo se pudo generar el reporte de calidad del lote")
        
        print("="*50)

def main():
    """Función principal del programa."""
    # Crear instancia del clasificador
    classifier = CoffeeBeanClassifier()
    
    # EJEMPLO DE USO CON IMAGEN LOCAL
    # Reemplazar con la ruta de tu imagen
    image_path = "./imagenes_para_analizar/grano1.jpg"
    
    # Verificar si la imagen existe
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen '{image_path}'")
        print("Por favor, coloca una imagen de granos de café en el directorio correcto")
        return
    
    # Analizar la imagen
    print("Iniciando análisis de imagen...")
    results = classifier.analyze_single_image(image_path)
    
    # Mostrar resumen
    classifier.print_summary(results)
    
    # Guardar resultados detallados
    output_file = classifier.save_results(results)
    
    if output_file:
        print(f"\nResultados detallados guardados en: {output_file}")

if __name__ == "__main__":
    main()