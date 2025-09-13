"""
Pacote de Análise de Postura

Este pacote contém e exporta todas as classes responsáveis por analisar
partes específicas da postura corporal.
- Agachamento: Detecta posição de agachamento.
- AlinhamentoOmbros: Detecta alinhamento dos ombros.
- CotoveloAcimaCabeca: Detecta cotovelos acima da cabeço.
- InclinacaoFrontal: Detecta inclinação frontal.
- InclinacaoLombar: Detecta inclinação lombar.
- InclinacaoParaTras: Detecta inclinação para trás.
- TorcaoPescoco: Detecta torção do pescoço.
"""

from .agachamento import Agachamento
from .alinhamento_ombros import AlinhamentoOmbros
from .cotovelo_acima_cabeca import CotoveloAcimaCabeca
from .inclinacao_frontal import InclinacaoFrontal
from .inclinacao_lombar import InclinacaoLombar
from .inclinacao_para_tras import InclinacaoParaTras
from .torcao_pescoco import TorcaoPescoco

__all__ = [
    'Agachamento',
    'AlinhamentoOmbros',
    'CotoveloAcimaCabeca',
    'InclinacaoFrontal',
    'InclinacaoLombar',
    'InclinacaoParaTras',
    'TorcaoPescoco'
]
