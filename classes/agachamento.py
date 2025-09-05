import cv2
import time

class Agachamento:
    def __init__(self, keypoints):
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.joelhoDireito = keypoints[14]
        self.joelhoEsquerdo = keypoints[13]
       
    def calcular_angulo_joelho(self, lado):
        if lado == "direito":
            quadril, joelho = self.quadrilDireito, self.joelhoDireito
        else:
            quadril, joelho = self.quadrilEsquerdo, self.joelhoEsquerdo

        
        return abs(joelho[1] - quadril[1])

    @staticmethod
    def detectar(cap, model):
        from tkinter import messagebox

        tempo_agachamento_incorreto = {}
        agachamento_iniciado = {}  

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Erro", "Erro ao capturar o frame da webcam.")
                break

            results = model(frame)
            pessoas = []

            for i, detection in enumerate(results[0].keypoints):
                for pessoaKey in detection.xy:
                    agachamento_incorreto = False
                    try:
                        agach = Agachamento(pessoaKey)
                        angulo_direito = agach.calcular_angulo_joelho("direito")
                        angulo_esquerdo = agach.calcular_angulo_joelho("esquerdo")
                       
                         
                        
                        if angulo_direito < 115 and angulo_esquerdo < 115 and angulo_direito > 0 and angulo_esquerdo > 0:
                            agachamento_iniciado[i] = True
                        elif angulo_direito >= 115 and angulo_esquerdo >= 115:
                            agachamento_iniciado[i] = False

                        if agachamento_iniciado.get(i, False) and (
                            angulo_direito < 40 or angulo_direito > 80 or
                            angulo_esquerdo < 40 or angulo_esquerdo > 80
                        ):
                            agachamento_incorreto = True
                    except Exception as e:
                        print(f"Erro ao calcular ângulo do joelho: {e}")

                    if agachamento_incorreto:
                        if i not in tempo_agachamento_incorreto:
                            # Se a pessoa estiver na posição incorreta, começa a contar o tempo
                            tempo_agachamento_incorreto[i] = time.time()
                        elif time.time() - tempo_agachamento_incorreto[i] >= 3:
                            # Mostra o alerta se o tempo incorreto exceder 3 segundos
                            posicao_texto = (50, 100 + i * 30)
                            cv2.putText(frame, f"Alerta! Pessoa {i + 1} com agachamento incorreto!", posicao_texto,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        # Remove o alerta se a pessoa voltar à posição correta
                        if i in tempo_agachamento_incorreto:
                            del tempo_agachamento_incorreto[i]

                    pessoas.append(agachamento_incorreto)
            


            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
     
        cap.release()