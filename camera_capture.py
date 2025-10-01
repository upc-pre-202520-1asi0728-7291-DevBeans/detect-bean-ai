"""
Módulo para captura de imágenes desde cámara.
ACTUALMENTE COMENTADO - No se ejecutará hasta que se descomente.
"""

import cv2
import numpy as np

class CameraCapture:
    """Clase para captura de imágenes desde cámara web."""
    
    def __init__(self, camera_index=0):
        """
        Inicializa la captura de cámara.
        
        Args:
            camera_index (int): Índice de la cámara a usar
        """
        self.camera_index = camera_index
        self.camera = None
    
    def initialize_camera(self):
        """
        Inicializa la cámara web.
        
        Returns:
            bool: True si la cámara se inicializó correctamente
        """
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            return self.camera.isOpened()
        except Exception as e:
            print(f"Error inicializando cámara: {e}")
            return False
    
    def capture_frame(self):
        """
        Captura un frame de la cámara.
        
        Returns:
            numpy.ndarray: Frame capturado o None si hay error
        """
        if self.camera is None or not self.camera.isOpened():
            print("Cámara no inicializada")
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            print("Error capturando frame")
            return None
    
    def capture_multiple_frames(self, num_frames=5, delay=1):
        """
        Captura múltiples frames con un delay entre ellos.
        
        Args:
            num_frames (int): Número de frames a capturar
            delay (int): Delay en segundos entre capturas
            
        Returns:
            list: Lista de frames capturados
        """
        frames = []
        for i in range(num_frames):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)
                print(f"Frame {i+1}/{num_frames} capturado")
            
            # Esperar antes de la siguiente captura
            cv2.waitKey(delay * 1000)
        
        return frames
    
    def release_camera(self):
        """Libera los recursos de la cámara."""
        if self.camera is not None:
            self.camera.release()
            cv2.destroyAllWindows()

"""
# EJEMPLO DE USO (COMENTADO - NO SE EJECUTARÁ)

def capture_from_camera_example():
    # Crear instancia de captura de cámara
    camera_capture = CameraCapture()
    
    # Inicializar cámara
    if camera_capture.initialize_camera():
        print("Cámara inicializada correctamente")
        
        # Capturar un frame
        frame = camera_capture.capture_frame()
        
        if frame is not None:
            # Guardar frame capturado
            cv2.imwrite('captured_bean.jpg', frame)
            print("Frame capturado y guardado como 'captured_bean.jpg'")
        
        # Liberar cámara
        camera_capture.release_camera()
    else:
        print("No se pudo inicializar la cámara")

# Esta función está comentada y no se ejecutará
# capture_from_camera_example()
"""