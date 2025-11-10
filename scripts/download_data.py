import os
from dotenv import load_dotenv

# --- PASO CRÍTICO: CARGAR VARIABLES ANTES DE CUALQUIER OTRA COSA ---
# Esto hace que KAGGLE_USERNAME y KAGGLE_KEY estén disponibles
# como variables de entorno al importar el módulo 'kaggle'.
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_NAME = 'gpiosenka/coffee-bean-dataset-resized-224-x-224'
DATA_DIR = 'data/coffee_beans'


def download_kaggle_dataset_api():
    """Descarga el dataset usando la API de Python de Kaggle."""
    print("--- 1. Descargando Dataset de Kaggle (Usando credenciales de .env) ---")

    # 1. Crear directorios
    os.makedirs(DATA_DIR, exist_ok=True)

    # Verifica si las credenciales fueron cargadas
    if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
        print("\nERROR: Las variables KAGGLE_USERNAME o KAGGLE_KEY no están cargadas.")
        print("Asegurarse de que 'python-dotenv' esté instalado y que el archivo .env esté en el directorio raíz del proyecto.")
        return

    try:
        # La autenticación ya ocurrió implícitamente al importar 'kaggle',
        # pero la llamamos explícitamente para mayor claridad si fuera necesario,
        # aunque el error ya fue manejado por las variables de entorno.
        api = KaggleApi()
        api.authenticate() # Esta llamada ahora debería ser redundante/exitosa si las variables están cargadas.

        # 2. Descargar y descomprimir
        api.dataset_download_files(
            dataset=DATASET_NAME,
            path=DATA_DIR,
            unzip=True
        )

        print(f"Dataset descargado y extraído con éxito en: {DATA_DIR}")

    except Exception as e:
        print(f"\nERROR DE LA API DE KAGGLE: Falló la descarga.")
        print("Asegurarse de que el usuario 'abrahamayquipa' tenga permisos para descargar este dataset.")
        print("Detalle del error:", e)

if __name__ == '__main__':
    download_kaggle_dataset_api()