class CotoveloAcimaCabeca:
    def __init__(self, keypoints):
        self.cotoveloDireito = keypoints[8]
        self.cotoveloEsquerdo = keypoints[7]
        self.nariz = keypoints[0]
    def calcular(self):
        return (self.cotoveloDireito[1] < self.nariz[1]) or (self.cotoveloEsquerdo[1] < self.nariz[1])

