import cv2
from tkinter import messagebox
from ultralytics import YOLO

# Classe para verificar alinhamento de ombros usando pontos-chave de pose
class alinhamentoOmbros:
    def __init__(self, keypoints):
        self.ombro_direito = keypoints[6]
        self.ombro_esquerdo = keypoints[5]

    def verificar(self, limite_tolerancia=15):
        # Calcula a diferença na altura entre os ombros
        diferenca_altura = abs(self.ombro_direito[1] - self.ombro_esquerdo[1])
        return diferenca_altura > limite_tolerancia

    @staticmethod
    def iniciar_detecao(cap, modelo_yolo):
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Erro", "Falha ao acessar a câmera.")
                break
            
            # Realiza a detecção de pose no frame
            resultados = modelo_yolo(frame)
            estados_ombros = []

            for pose in resultados[0].keypoints:
                for keypoints in pose.xy:
                    desalinhado = False
                    try:
                        desalinhado = alinhamentoOmbros(keypoints).verificar()
                    except Exception as erro:
                        print(f"Erro ao verificar alinhamento: {erro}")
                    estados_ombros.append(desalinhado)

            for indice, desalinhado in enumerate(estados_ombros):
                posicao_mensagem = (50, 100 + indice * 30)
                if desalinhado:
                    cv2.putText(frame, f"Pessoa {indice + 1}: Ombros desalinhados!", posicao_mensagem, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"Pessoa {indice + 1}: Ombros alinhados!", posicao_mensagem, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            frame_anotado = resultados[0].plot()
            cv2.imshow("Detecção de Alinhamento de Ombros", frame_anotado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

