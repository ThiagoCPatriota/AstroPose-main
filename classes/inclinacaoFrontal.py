class inclinacaoFrontal:
    def __init__(self, keypoints):
        self.ombroDireito = keypoints[6]
        self.ombroEsquerdo = keypoints[5]
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.nariz = keypoints[0]
    
    def calcular(self):
        # Calcula o centro dos ombros e o centro dos quadris
        centroOmbrosY = (self.ombroDireito[1] + self.ombroEsquerdo[1]) / 2
        centroQuadrisY = (self.quadrilDireito[1] + self.quadrilEsquerdo[1]) / 2

        # Calcula a altura do corpo (da cabeça até o quadril)
        alturaCorpo = self.nariz[1] - centroQuadrisY

        # Calcula a inclinação proporcional entre os ombros e os quadris
        inclinacaoFrontal = (centroOmbrosY - centroQuadrisY) / alturaCorpo

        return inclinacaoFrontal
    
    @staticmethod
    def detectar(cap, model):
        from tkinter import messagebox
        import cv2
        import time
        tempo_inclinacao = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Erro", "Erro ao capturar o frame da webcam.")
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            pessoas = []

            for i, (box, keypoints) in enumerate(zip(results[0].boxes, results[0].keypoints)):
                for pessoaKey in keypoints.xy:
                    
                    try:
                        incFront = inclinacaoFrontal(pessoaKey).calcular()
                        if incFront > 0.85:  # Ajuste o valor conforme necessário
                            torceu = True
                    except Exception as e:
                        print(f"Erro ao calcular inclinação: {e}")
                        incFront = 0

                if torceu:
                    if i not in tempo_inclinacao:
                        tempo_inclinacao[i] = time.time()
                    elif time.time() - tempo_inclinacao[i] >= 2:
                        cor_borda = (0, 0, 255)  # Vermelho para inclinação
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), cor_borda, 4)
                         

                else:
                    if i in tempo_inclinacao:
                        del tempo_inclinacao[i]

            
            cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
    

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
