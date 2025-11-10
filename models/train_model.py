import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMAGE_SIZE, BEAN_CATEGORIES, MODEL_PATHS

# --- CONFIGURACIÓN ---
DATA_DIR = 'scripts/data/coffee_beans/train' # Ruta que contiene las imágenes de entrenamiento
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)  # (224, 224, 3)
NUM_CLASSES = len(BEAN_CATEGORIES)
BATCH_SIZE = 32
EPOCHS = 5  # Usamos pocas épocas por ser un ejemplo rápido; un modelo real requiere más.


def build_cnn_model():
    """Define y construye una CNN simple."""
    model = Sequential([
        # Primera capa convolucional y de pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(2, 2),

        # Segunda capa convolucional y de pooling
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Tercera capa para extraer más características
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Aplanar para la capa densa
        Flatten(),

        # Capa densa con Dropout para evitar overfitting
        Dense(512, activation='relu'),
        Dropout(0.5),

        # Capa de salida: 5 neuronas (tipos de defecto) y Softmax
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compilación: Optimizador Adam, pérdida categórica para clasificación
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def train_and_save_model():
    """Prepara datos, entrena el modelo y lo guarda."""
    print("\n--- 2. Preparando Data Generators ---")

    # 1. Definir Data Augmentation (Técnica para mejorar la generalización)
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalización de píxeles
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validación
    )

    # 2. Generador de Entrenamiento
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # 3. Generador de Validación
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 4. Construir y entrenar
    model = build_cnn_model()

    print("\n--- 3. Iniciando Entrenamiento ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # 5. Guardar el modelo en la ruta definida en config.py
    os.makedirs(os.path.dirname(MODEL_PATHS['defect_detector']), exist_ok=True)
    model.save(MODEL_PATHS['defect_detector'])
    print(f"\n--- 4. Modelo CNN guardado exitosamente en: {MODEL_PATHS['defect_detector']} ---")


if __name__ == '__main__':
    # Verificar si ya existe la carpeta de datos
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) < 2:
        print("Datos no encontrados. Ejecuta 'python download_data.py' primero.")
    else:
        train_and_save_model()