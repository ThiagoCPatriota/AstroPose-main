class InclinacaoParaTras:
    def __init__(self, keypoints):
        self.nariz = keypoints[0]
        self.ombroDireito = keypoints[6]
        self.ombroEsquerdo = keypoints[5]
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
    def calcular(self):
        ombro_medio_x = (self.ombroDireito[0] + self.ombroEsquerdo[0]) / 2.0
        quadril_medio_x = (self.quadrilDireito[0] + self.quadrilEsquerdo[0]) / 2.0
        nariz_x = self.nariz[0]
        ombros_inclinados = ombro_medio_x < (quadril_medio_x - 50)
        cabeca_inclinada = nariz_x < (quadril_medio_x - 30)
        return bool(ombros_inclinados and cabeca_inclinada)