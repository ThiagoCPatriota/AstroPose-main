import tkinter as tk
import cv2
from src.core.detector import PoseDetector
from tkinter import messagebox

detector = None

def iniciar_camera_e_detector(model_path="models/yolov8n-pose.pt", width=640, height=480, interval_ms=150, force_cpu=True):
    global detector
    # inicializa detector com configurações mais leves
    detector = PoseDetector(
        camera_index=0,
        model_path=model_path,
        width=width,
        height=height,
        frame_interval_ms=interval_ms,
        draw_annotations=False,
        force_cpu=force_cpu,
        face_recognition_enabled=False,
    )

def liberar_camera():
    global detector
    if detector:
        detector.liberarRecursos()
        detector = None

def _loop_detect(tipo):
    # Apenas um loop de demonstração que chama detectar_pose até o usuário parar a janela
    global detector
    if not detector:
        messagebox.showerror("Erro", "Detector não inicializado.")
        return

    try:
        while True:
            mensagens, annotated_frame, keypoints = detector.detectar_pose()
            # mostramos janela com frame anotado (redimensionado)
            if annotated_frame is not None:
                cv2.imshow(f"Detecção - {tipo}", annotated_frame)
            if mensagens:
                print("Mensagens:", mensagens)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("Erro no loop de detecção:", e)
    finally:
        cv2.destroyAllWindows()

def IniciarDetecPadrao():
    liberar_camera()
    iniciar_camera_e_detector(model_path="../models/yolov8n-pose.pt", width=640, height=480, interval_ms=150, force_cpu=True)
    _loop_detect("Padrão")
    liberar_camera()

def IniciarDetecRapida():
    liberar_camera()
    # resolução menor e intervalo maior -> mais fps
    iniciar_camera_e_detector(model_path="../models/yolov8n-pose.pt", width=320, height=240, interval_ms=80, force_cpu=True)
    _loop_detect("Rápida")
    liberar_camera()

# Configurar a interface gráfica básica
root = tk.Tk()
root.title("Detecção de Pose (Modo Sem Placa)")
root.geometry("320x160")

start_button = tk.Button(root, text="Iniciar (padrão)", command=IniciarDetecPadrao)
start_button.pack(pady=10)

start_button2 = tk.Button(root, text="Iniciar rápida (menor)", command=IniciarDetecRapida)
start_button2.pack(pady=10)

close_button = tk.Button(root, text="Fechar", command=lambda: (liberar_camera(), root.destroy()))
close_button.pack(pady=10)

root.mainloop()
