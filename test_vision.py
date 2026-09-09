import os
import cv2
from ultralytics import YOLO

class MotorVisionScanEats:
    # Cambiamos a la versión 's' (Small) para mayor precisión en la detección
    def __init__(self, ruta_video="video.mp4", modelo="yolov8s.pt"):
        if not os.path.exists(ruta_video):
            raise FileNotFoundError(f"No se encuentra el archivo '{ruta_video}' en: {os.getcwd()}")
        
        print("Cargando modelo YOLOv8s (Alta precisión)...")
        self.modelo = YOLO(modelo)
        self.ruta_video = ruta_video

    def extraer_torso_superior(self, coordenadas):
        x1, y1, x2, y2 = map(int, coordenadas)
        altura_total = y2 - y1
        altura_torso = int(altura_total * 0.25)
        y2_superior = y1 + altura_torso
        return x1, y1, x2, y2_superior

    def procesar_flujo(self):
        cap = cv2.VideoCapture(self.ruta_video)
        if not cap.isOpened():
            raise ValueError(f"OpenCV no pudo abrir '{self.ruta_video}'. Verifica el formato.")

        print("Iniciando procesamiento de video...")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            # Usamos el fotograma original (sin reducir) y le decimos a YOLO 
            # que acepte detecciones desde el 25% de confianza (conf=0.25)
            resultados = self.modelo(frame, classes=[0], conf=0.25, verbose=False)
            frame_anotado = frame.copy()

            for box in resultados[0].boxes:
                x1, y1, x2, y2_sup = self.extraer_torso_superior(box.xyxy[0])
                confianza = float(box.conf[0])
                
                # Dibujamos en verde para diferenciar esta nueva versión
                cv2.rectangle(frame_anotado, (x1, y1), (x2, y2_sup), (0, 255, 0), 2)
                cv2.putText(frame_anotado, f"Torso {confianza:.2f}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Achicamos la imagen ÚNICAMENTE para mostrarla en tu monitor, 
            # el modelo de IA ya procesó la imagen grande con todos los detalles
            vista_pantalla = cv2.resize(frame_anotado, (800, 600))
            cv2.imshow("ScanEats - Motor de Vision (Alta Precision)", vista_pantalla)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Proceso finalizado.")

if __name__ == "__main__":
    motor = MotorVisionScanEats()
    motor.procesar_flujo()