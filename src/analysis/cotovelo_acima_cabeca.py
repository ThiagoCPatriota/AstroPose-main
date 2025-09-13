class CotoveloAcimaCabeca:
    """
    Verifica se um ou ambos os cotovelos estão levantados acima da linha da cabeça (nariz).
    """
    def __init__(self, keypoints):
        """
        Inicializa a análise com os keypoints de uma pessoa.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) dos keypoints da pose.
        """
        self.cotoveloDireito = keypoints[8]
        self.cotoveloEsquerdo = keypoints[7]
        self.nariz = keypoints[0]

    def calcular(self):
        """
        Verifica se a coordenada Y de algum cotovelo é menor (mais alta) que a do nariz.

        Returns:
            bool: True se um cotovelo estiver acima da cabeça, False caso contrário.
        """
        return (self.cotoveloDireito[1] < self.nariz[1]) or (self.cotoveloEsquerdo[1] < self.nariz[1])

