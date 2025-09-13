class InclinacaoParaTras:
    """
    Verifica se o corpo está excessivamente inclinado para trás.
    """
    def __init__(self, keypoints):
        """
        Inicializa a análise com os keypoints de uma pessoa.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) dos keypoints da pose.
        """
        self.nariz = keypoints[0]
        self.ombroDireito = keypoints[6]
        self.ombroEsquerdo = keypoints[5]
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]

    def calcular(self):
        """
        Verifica se a cabeça e os ombros estão muito atrás da linha do quadril.

        Returns:
            bool: True se houver inclinação para trás, False caso contrário.
        """
        ombro_medio_x = (self.ombroDireito[0] + self.ombroEsquerdo[0]) / 2.0
        quadril_medio_x = (self.quadrilDireito[0] + self.quadrilEsquerdo[0]) / 2.0
        nariz_x = self.nariz[0]
        ombros_inclinados = ombro_medio_x < (quadril_medio_x - 50)
        cabeca_inclinada = nariz_x < (quadril_medio_x - 30)
        return bool(ombros_inclinados and cabeca_inclinada)