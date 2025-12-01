import math


class TorcaoPescoco:
    """
    Analisa a torção do pescoço com base na visibilidade das orelhas e na
    distância do nariz aos ombros.
    """
    def __init__(self, keypoints):
        """
        Inicializa a análise com os keypoints de uma pessoa.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) dos keypoints da pose.
        """
        self.nariz = keypoints[0]
        self.orelhaDi = keypoints[4]
        self.orelhaEs = keypoints[3]
        self.ombroEs = keypoints[5]
        self.ombroDi = keypoints[6]

    def calcular(self):
        """
        Calcula se há uma torção de pescoço significativa.

        Returns:
            bool: True se houver torção, False caso contrário.
        """
        ombroDiExiste = True
        ombroEsExiste = True
        orelhaDiExiste = True
        orelhaEsExiste = True
        margem_percentual = 15
        try:
            distanciaOmbroDi = math.hypot(self.nariz[0] - self.ombroDi[0], self.nariz[1] - self.ombroDi[1])
        except Exception:
            ombroDiExiste = False
            distanciaOmbroDi = 0.0
        try:
            distanciaOmbroEs = math.hypot(self.nariz[0] - self.ombroEs[0], self.nariz[1] - self.ombroEs[1])
        except Exception:
            ombroEsExiste = False
            distanciaOmbroEs = 0.0
        try:
            distanciaOrelhaDi = math.hypot(self.nariz[0] - self.orelhaDi[0], self.nariz[1] - self.orelhaDi[1])
        except Exception:
            orelhaDiExiste = False
            distanciaOrelhaDi = 0.0
        try:
            distanciaOrelhaEs = math.hypot(self.nariz[0] - self.orelhaEs[0], self.nariz[1] - self.orelhaEs[1])
        except Exception:
            orelhaEsExiste = False
            distanciaOrelhaEs = 0.0

        if ombroDiExiste and ombroEsExiste:
            margem = margem_percentual / 100.0
            diferenca_permitida = distanciaOmbroEs * margem
            return abs(distanciaOmbroDi - distanciaOmbroEs) > diferenca_permitida

        if ombroDiExiste and orelhaDiExiste and not orelhaEsExiste:
            if distanciaOrelhaDi > distanciaOmbroDi:
                return True
        if ombroEsExiste and orelhaEsExiste and not orelhaDiExiste:
            if distanciaOrelhaEs > distanciaOmbroEs:
                return True
        return False