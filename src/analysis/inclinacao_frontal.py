class InclinacaoFrontal:
    """
    Calcula a inclinação frontal do tronco.
    """
    def __init__(self, keypoints):
        """
        Inicializa a análise com os keypoints de uma pessoa.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) dos keypoints da pose.
        """
        self.ombroDireito = keypoints[6]
        self.ombroEsquerdo = keypoints[5]
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.nariz = keypoints[0]

    def calcular(self):
        """
        Calcula a inclinação proporcional entre os ombros e os quadris.

        Returns:
            float: Um valor que representa a inclinação frontal.
        """
        centroOmbrosY = (self.ombroDireito[1] + self.ombroEsquerdo[1]) / 2.0
        centroQuadrisY = (self.quadrilDireito[1] + self.quadrilEsquerdo[1]) / 2.0
        alturaCorpo = self.nariz[1] - centroQuadrisY
        if abs(alturaCorpo) < 1e-6:
            return 0.0
        return (centroOmbrosY - centroQuadrisY) / alturaCorpo
