"""
Pacote Core do AstroPose

Este pacote contém as funcionalidades centrais de detecção e análise.
Ele exporta as seguintes classes para facilitar o acesso:
- PoseDetector: A classe principal para detecção de pose.
- ReconhecimentoFacial: A classe de gerenciamento de reconhecimento facial.
"""

from .detector import PoseDetector
from .reconhecimento_facial import ReconhecimentoFacial

__all__ = [
    'AstroPoseMainWindow'
]