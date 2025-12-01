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
