import numpy as np

class InclinacaoLombar:
    """
    Analisa a inclinação lateral do tronco, verificando os ângulos dos ombros e quadris.
    """
    def __init__(self, keypoints):
        """
        Inicializa a análise com os keypoints de uma pessoa.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) dos keypoints da pose.
        """
        self.ombro_esq = keypoints[5]
        self.ombro_dir = keypoints[6]
        self.quadril_esq = keypoints[11]
        self.quadril_dir = keypoints[12]

    def calcular(self):
        """
        Calcula se o ângulo de inclinação dos ombros ou dos quadris está em uma faixa de risco.

        Returns:
            bool: True se houver risco de inclinação lombar, False caso contrário.
        """
        ANGULO_RISCO_MIN = 35
        ANGULO_RISCO_MAX = 75
        dy_ombro = self.ombro_dir[1] - self.ombro_esq[1]
        dx_ombro = self.ombro_dir[0] - self.ombro_esq[0]
        dy_quadril = self.quadril_dir[1] - self.quadril_esq[1]
        dx_quadril = self.quadril_dir[0] - self.quadril_esq[0]
        angulo_ombro = abs(np.degrees(np.arctan2(dy_ombro, dx_ombro)))
        angulo_quadril = abs(np.degrees(np.arctan2(dy_quadril, dx_quadril)))
        risco_ombro = ANGULO_RISCO_MIN <= angulo_ombro <= ANGULO_RISCO_MAX
        risco_quadril = ANGULO_RISCO_MIN <= angulo_quadril <= ANGULO_RISCO_MAX
        return risco_ombro or risco_quadril