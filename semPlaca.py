import tkinter as tk
import cv2
from ultralytics import YOLO
from classes.inclinacaoFrontal import inclinacaoFrontal
from classes.alinhamentoOmbros import alinhamentoOmbros
from classes.agachamento import Agachamento  
from tkinter import messagebox


# Carregar o modelo de detecção de pose do YOLOv8
model = YOLO('yolov8n-pose.pt')

# Variável global para capturar o vídeo da webcam
cap = None

def iniciar_camera():
    global cap
    cap = cv2.VideoCapture(0)  # Pode mudar o índice se houver mais de uma câmera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def liberar_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def IniciarDetecFront():
    liberar_camera()
    iniciar_camera()
    
    if not cap or not cap.isOpened():
        messagebox.showerror("Erro", "Erro ao acessar a webcam.")
        return

    inclinacaoFrontal.detectar(cap, model)
    liberar_camera()  # Liberar a câmera após a detecção

def IniciarDetecOmbros():
    liberar_camera()
    iniciar_camera()

    if not cap or not cap.isOpened():
        messagebox.showerror("Erro", "Erro ao acessar a webcam.")
        return

    alinhamentoOmbros.iniciar_detecao(cap, model)
    liberar_camera()  # Liberar a câmera após a detecção

def IniciarDetecAgachamento():
    liberar_camera()
    iniciar_camera()

    if not cap or not cap.isOpened():
        messagebox.showerror("Erro", "Erro ao acessar a webcam.")
        return

    Agachamento.detectar(cap, model)  
    liberar_camera()  

# Configurar a interface gráfica
root = tk.Tk()
root.title("Detecção de Pose com YOLOv8")
root.geometry("300x150")

# Adicionar os botões
start_button = tk.Button(root, text="Detectar Inclinação Frontal", command=IniciarDetecFront)
start_button.pack(pady=10)

start_button2 = tk.Button(root, text="Detectar Alinhamento de Ombros", command=IniciarDetecOmbros)
start_button2.pack(pady=10)

start_button3 = tk.Button(root, text="Detectar Agachamento", command=IniciarDetecAgachamento)
start_button3.pack(pady=10)

# Iniciar a interface gráfica
root.mainloop()
