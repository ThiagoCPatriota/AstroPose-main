class AlinhamentoOmbros:
    """
    Verifica o alinhamento vertical dos ombros para detectar desníveis.
    """
    def __init__(self, keypoints):
        """
        Inicializa a análise com os keypoints de uma pessoa.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) dos keypoints da pose.
        """
        self.ombro_direito = keypoints[6]
        self.ombro_esquerdo = keypoints[5]

    def verificar(self, limite_tolerancia=15):
        """
        Verifica se a diferença de altura entre os ombros excede um limite.

        Args:
            limite_tolerancia (int): A diferença máxima de pixels na vertical
                                     permitida antes de considerar os ombros desalinhados.

        Returns:
            bool: True se os ombros estiverem desalinhados, False caso contrário.
        """
        diferenca_altura = abs(self.ombro_direito[1] - self.ombro_esquerdo[1])
        return diferenca_altura > limite_tolerancia