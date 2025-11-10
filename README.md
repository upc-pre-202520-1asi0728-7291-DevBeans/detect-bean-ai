# BeanDetect AI: Clasificador de Calidad de Granos de Café

Este proyecto es el motor de clasificación de `DevBeans`, una aplicación de Python que utiliza Visión por Computadora (CV) y una Red Neuronal Convolucional (CNN) para analizar imágenes de granos de café y asignarles una categoría de calidad.

El sistema segmenta granos individuales de una imagen, ejecuta un modelo de IA entrenado para determinar la clase de color/tueste, y luego aplica una lógica de negocio para asignar una puntuación y una categoría final (Specialty, Premium, A, B, C).

## Tecnologías y Dependencias

  * **Python 3.11+**
  * **TensorFlow (Keras):** Para construir, entrenar y ejecutar el modelo CNN.
  * **OpenCV (`opencv-python`):** Para todas las tareas de procesamiento de imágenes, segmentación y extracción de características.
  * **Scikit-image:** Para la extracción de características de textura (LBP).
  * **Numpy:** Para la manipulación de matrices de imágenes.
  * **Kaggle:** Para utilizar el dataset de entrenamiento.
  * **Dotenv:** Para gestionar variables de entorno (credenciales de Kaggle).
  * **Git LFS:** (Almacenamiento de Archivos Grandes) Requerido para manejar el archivo del modelo CNN (`.h5`).

-----

## Configuración y Ejecución

Esta guía es para un desarrollador que quiera clonar el repositorio y **ejecutar la clasificación** en su máquina local.

### 1\. Configuración de Git LFS (Paso Crítico)

El modelo de IA (`.h5`) es demasiado grande para Git (508 MiB), por lo que usamos Git LFS. Debes "activar" LFS en tu máquina **una sola vez**.

```bash
# 1. Instala el cliente de Git LFS desde Git Bash mediante el comando mostrado
git lfs install
# 2. Si lo anterior no funciona, instala Git LFS desde https://git-lfs.github.com/
```

### 2\. Clonar y Configurar el Entorno

```bash
# 1. Clona el repositorio

# 2. Instala las dependencias
# Git LFS descargará automáticamente el archivo .h5 durante este 'pull' o 'clone'
git lfs pull
pip install -r requirements.txt
```

### 3\. Ejecutar la Clasificación

¡Ya estás listo para clasificar\!

```bash
# Coloca las imágenes que quieres analizar en la carpeta 'imagenes_para_analizar/'
# (El sistema ya cuenta con algunas imágenes de prueba como 'granoverde.png' o 'grano1.jpg')

# Ejecuta el programa principal
python main.py
```

### 4\. Revisar los Resultados

El script imprimirá un resumen en tu terminal. También generará un archivo `.json` detallado (ej. `coffee_analysis_...json`) con la siguiente información clave por cada grano:

  * `color_percentages`: La confianza de la CNN para cada clase de color (Dark, Green, Light, Medium).
  * `quality_assessment`: La puntuación final y la categoría de calidad asignada (A, B, C, etc.).

-----

## 🧠 Cómo Reentrenar el Modelo de IA (Avanzado)

Si deseas mejorar el modelo CNN (ej. entrenar con más épocas o más datos), el proceso es diferente.

### 1\. Requisitos Previos

Necesitarás credenciales de Kaggle para descargar el *dataset*.

1.  Crea un archivo `.env` en la raíz del proyecto.
2.  Añade tus credenciales de Kaggle (obtenidas de tu `kaggle.json`):
    ```.env
    KAGGLE_USERNAME="tu-usuario"
    KAGGLE_KEY="tu-llave-api"
    ```

### 2\. Ejecutar el Flujo de Entrenamiento

```bash
# 1. Descarga el dataset (creará la carpeta 'scripts/data/')
python -m scripts.download_data

# 2. Ejecuta el entrenamiento (Esto tomará varios minutos)
# Sobrescribirá 'models/defect_detector.h5' con la nueva versión
python -m models.train_model
```

### 3\. Subir el Nuevo Modelo

Dado que el archivo `.h5` está rastreado por LFS, simplemente haz `commit` y `push` del archivo actualizado.

```bash
git add models/defect_detector.h5
git commit -m "refactor(IA): Re-trained CNN model with 50 epochs"
git push
```