import cv2
from ultralytics import YOLO
import torch
import time
import math
import numpy as np

class InclinacaoLombar:
    def __init__(self, keypoints):
        # Inicializa os pontos dos ombros e quadris
        self.ombro_esq = keypoints[5]  # Ombro esquerdo (índice 5)
        self.ombro_dir = keypoints[6]  # Ombro direito (índice 6)
        self.quadril_esq = keypoints[11]  # Quadril esquerdo (índice 11)
        self.quadril_dir = keypoints[12]  # Quadril direito (índice 12)

    def calcular(self):
        # Definição dos ângulos mínimos e máximos para risco de inclinação
        ANGULO_RISCO_MIN = 35
        ANGULO_RISCO_MAX = 75

        # Calcula a diferença de posição entre os ombros
        dy_ombro = self.ombro_dir[1] - self.ombro_esq[1]
        dx_ombro = self.ombro_dir[0] - self.ombro_esq[0]

        # Calcula a diferença de posição entre os quadris
        dy_quadril = self.quadril_dir[1] - self.quadril_esq[1]
        dx_quadril = self.quadril_dir[0] - self.quadril_esq[0]

        # Calcula o ângulo da inclinação da linha entre os ombros e os quadris
        angulo_ombro = abs(np.degrees(np.arctan2(dy_ombro, dx_ombro)))
        angulo_quadril = abs(np.degrees(np.arctan2(dy_quadril, dx_quadril)))

        # Verifica se os ângulos estão dentro da faixa de risco
        risco_ombro = ANGULO_RISCO_MIN <= angulo_ombro <= ANGULO_RISCO_MAX
        risco_quadril = ANGULO_RISCO_MIN <= angulo_quadril <= ANGULO_RISCO_MAX

        # Retorna True se qualquer uma das inclinações estiver na faixa de risco
        return risco_ombro or risco_quadril

model = YOLO('yolov8n-pose.pt')

@staticmethod
def deteccao(cap, model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Realizar a detecção de pose
        results = model(frame)
        pessoas = []
        
        for detection in results[0].keypoints:
            for pessoaKey in detection.xy:
                try:
                    # Calcular inclinações do quadril e ombro para cada pessoa
                    inclinacao_corpo = InclinacaoLombar(pessoaKey)
                    angulo_quadril, mensagem_quadril = inclinacao_corpo.calcular_inclinacao_quadril()
                    angulo_ombro, mensagem_ombro = inclinacao_corpo.calcular_inclinacao_ombro()
                    
                    # Armazenar status de inclinação do quadril e ombro
                    pessoas.append((mensagem_quadril, mensagem_ombro, angulo_quadril, angulo_ombro))
                except Exception as e:
                    print(f"Erro ao calcular inclinação: {e}")
        
        # Exibir mensagens no vídeo
        for j, (mensagem_quadril, mensagem_ombro, angulo_quadril, angulo_ombro) in enumerate(pessoas):
            posicao_texto = (50, 100 + j * 60)
            cv2.putText(frame, f"Pessoa {j + 1}: {mensagem_quadril} ({angulo_quadril:.2f}°)", posicao_texto, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Pessoa {j + 1}: {mensagem_ombro} ({angulo_ombro:.2f}°)", (posicao_texto[0], posicao_texto[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Exibir o frame anotado
        annotated_frame = results[0].plot()
        cv2.imshow("Detecção de Inclinação - Quadril e Ombros", annotated_frame)
        
        # Pressionar 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar o vídeo e fechar as janelas
    cap.release()
    cv2.destroyAllWindows()
