import cv2
from ultralytics import YOLO
import torch
import time
import math
import numpy as np
from reconhecimento_facial import ReconhecimentoFacial

def _to_np(x):
    """Converte tensor (CPU/CUDA) para numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class PoseDetector:
    def get_keypoints(self):
        return getattr(self, "_last_keypoints_ui", None)

    def get_issue_code(self):
        return getattr(self, "_last_issue_code", "OK")

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n-pose.pt").to(self.device)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.tempo_cotovelo = {}
        self.tempo_inclinacao = {}
        self.tempo_inclinacao_frontal = {}
        self.tempoTorcao = {}
        self.tempoLombar = {}
        self.tempo_agachamento_incorreto = {}

        self._last_keypoints_ui = None
        self._last_issue_code = "OK"

        # Novo: Reconhecimento Facial
        self.face_recognizer = ReconhecimentoFacial()
        self.cadastro_ativo = False
        self.nome_cadastro = None

        if not self.cap.isOpened():
            raise Exception("Erro ao acessar a webcam.")

    def iniciar_cadastro_astronauta(self, nome=None):
        """Inicia o modo de cadastro de astronauta"""
        self.cadastro_ativo = True
        self.nome_cadastro = nome
        return f"Modo cadastro ativo para: {nome if nome else 'novo astronauta'}"

    def finalizar_cadastro_astronauta(self, frame):
        """Finaliza o cadastro com o frame atual"""
        if self.cadastro_ativo:
            success = self.face_recognizer.cadastrar_astronauta(frame, self.nome_cadastro)
            self.cadastro_ativo = False
            self.nome_cadastro = None
            return success
        return False

    def esta_em_modo_cadastro(self):
        """Verifica se está em modo de cadastro"""
        return self.cadastro_ativo

    def detectar_pose(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Erro ao capturar o frame da webcam.")

        with torch.no_grad():
            results = self.model(frame, verbose=False)
        res0 = results[0]
        annotated_frame = res0.plot()
        mensagens = []

        # ---------- Reconhecimento Facial ----------
        face_results = self.face_recognizer.reconhecer_rostos(frame)
        for name, (left, top, right, bottom), confidence in face_results:
            if confidence > 0.7:
                color = (0, 255, 0)
                label = f"{name} ({confidence:.0%})"
            elif confidence > 0.5:
                color = (0, 255, 255)
                label = f"{name}? ({confidence:.0%})"
            else:
                color = (0, 0, 255)
                label = "Desconhecido"
            
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(annotated_frame, label, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if confidence > 0.7 and name != "Desconhecido":
                mensagens.append(f"Astronauta {name} detectado!")

        # Modo Cadastro
        if self.cadastro_ativo:
            cv2.putText(annotated_frame, "MODO CADASTRO - Posicione o rosto", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Nome: {self.nome_cadastro if self.nome_cadastro else 'Digite nome...'}", 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---------- Detecção de Pose (código original) ----------
        n = len(res0.boxes) if res0.boxes is not None else 0
        if n > 0:
            xy_all = _to_np(res0.keypoints.xy)
            try:
                conf_all = _to_np(res0.keypoints.conf)
            except AttributeError:
                conf_all = np.ones((n, 17), dtype=np.float32)
            xyxy_all = _to_np(res0.boxes.xyxy)

            areas = (xyxy_all[:, 2] - xyxy_all[:, 0]) * (xyxy_all[:, 3] - xyxy_all[:, 1])
            idx_max = int(np.argmax(areas))

            kp0 = [(float(x), float(y), float(c)) for (x, y), c in zip(xy_all[idx_max], conf_all[idx_max])]
            self._last_keypoints_ui = kp0

            for i in range(n):
                pessoaKey = xy_all[i]
                xyxy = xyxy_all[i]
                box_tuple = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])

                cotoveloAcima = False
                inclinadoParaTras = False
                inclinadoFrontal = False
                torceu = False
                inclinaLombar = False
                agachamento_incorreto = False
                agachamento_iniciado = {}
                desalinhado = False

                # Checks originais (mantidos)
                try:
                    if CotoveloAcimaCabeca(pessoaKey).calcular():
                        cotoveloAcima = True
                except Exception as e:
                    print(f"Erro ao calcular cotovelo acima da cabeça: {e}")

                try:
                    if InclinacaoLombar(pessoaKey).calcular():
                        inclinaLombar = True
                except Exception as e:
                    print(f"Erro ao calcular Inclinação lombar: {e}")

                try:
                    if InclinacaoParaTras(pessoaKey).calcular():
                        inclinadoParaTras = True
                except Exception as e:
                    print(f"Erro ao calcular Inclinação para trás: {e}")

                try:
                    val = inclinacaoFrontal(pessoaKey).calcular()
                    if val > 0.85:
                        inclinadoFrontal = True
                except Exception as e:
                    print(f"Erro ao calcular inclinação frontal: {e}")

                try:
                    desalinhado = AlinhamentoOmbros(pessoaKey).verificar()
                    if desalinhado:
                        mensagens.append(f"Pessoa {i + 1} com ombros desalinhados!")
                        self.bordaVermelha(box_tuple, annotated_frame)
                except Exception as e:
                    print(f"Erro ao calcular alinhamento de ombros: {e}")

                try:
                    agach = Agachamento(pessoaKey)
                    angulo_direito = agach.calcular_angulo_joelho("direito")
                    angulo_esquerdo = agach.calcular_angulo_joelho("esquerdo")
                    if angulo_direito < 115 and angulo_esquerdo < 155 and angulo_direito > 0 and angulo_esquerdo > 0:
                        agachamento_iniciado[i] = True
                    elif angulo_direito >= 115 and angulo_esquerdo >= 115:
                        agachamento_iniciado[i] = False

                    if agachamento_iniciado.get(i, False) and (
                        angulo_direito < 40 or angulo_direito > 80 or angulo_esquerdo < 40 or angulo_esquerdo > 80
                    ):
                        agachamento_incorreto = True
                except Exception as e:
                    print(f"Erro ao calcular ângulo do joelho: {e}")

                # Temporizações originais (mantidas)
                if cotoveloAcima:
                    if i not in self.tempo_cotovelo:
                        self.tempo_cotovelo[i] = time.time()
                    elif time.time() - self.tempo_cotovelo[i] >= 3:
                        self.bordaVermelha(box_tuple, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} levantou o braço por mais de 3 segundos!")
                else:
                    self.tempo_cotovelo.pop(i, None)

                if inclinadoParaTras:
                    if i not in self.tempo_inclinacao:
                        self.tempo_inclinacao[i] = time.time()
                    elif time.time() - self.tempo_inclinacao[i] >= 3:
                        self.bordaVermelha(box_tuple, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está inclinada para trás por mais de 3 segundos!")
                else:
                    self.tempo_inclinacao.pop(i, None)

                if inclinadoFrontal:
                    if i not in self.tempo_inclinacao_frontal:
                        self.tempo_inclinacao_frontal[i] = time.time()
                    elif time.time() - self.tempo_inclinacao_frontal[i] >= 3:
                        self.bordaVermelha(box_tuple, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está inclinada frontalmente por mais de 3 segundos!")
                else:
                    self.tempo_inclinacao_frontal.pop(i, None)

                if inclinaLombar:
                    if i not in self.tempoLombar:
                        self.tempoLombar[i] = time.time()
                    elif time.time() - self.tempoLombar[i] >= 3:
                        self.bordaVermelha(box_tuple, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está com inclinação lombar excessiva por mais de 3 segundos!")
                else:
                    self.tempoLombar.pop(i, None)

                if agachamento_incorreto:
                    if i not in self.tempo_agachamento_incorreto:
                        self.tempo_agachamento_incorreto[i] = time.time()
                    elif time.time() - self.tempo_agachamento_incorreto[i] >= 3:
                        self.bordaVermelha(box_tuple, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está agachando incorretamente por mais de 3 segundos")
                else:
                    self.tempo_agachamento_incorreto.pop(i, None)

                try:
                    torceu = torcaoPescoco(pessoaKey).calcular()
                    if torceu:
                        if i not in self.tempoTorcao:
                            self.tempoTorcao[i] = time.time()
                        elif time.time() - self.tempoTorcao[i] >= 3:
                            self.bordaVermelha(box_tuple, annotated_frame)
                            mensagens.append(f"Alerta! Pessoa {i + 1} torceu o pescoço por mais de 3 segundos!")
                    else:
                        self.tempoTorcao.pop(i, None)
                except Exception as e:
                    print(f"Erro ao calcular torção: {e}")

                issue = "OK"
                if agachamento_incorreto or inclinaLombar:
                    issue = "LUMBAR"
                elif inclinadoFrontal:
                    issue = "FRONT_TILT"
                elif inclinadoParaTras:
                    issue = "BACK_TILT"
                elif desalinhado:
                    issue = "SHOULDER_MISALIGN"
                elif torceu:
                    issue = "NECK_TWIST"
                elif cotoveloAcima:
                    issue = "ELBOW_UP"

                if i == idx_max:
                    self._last_issue_code = issue

        else:
            self._last_keypoints_ui = None
            self._last_issue_code = "OK"

        return mensagens, annotated_frame, self._last_keypoints_ui

    def liberarRecursos(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def bordaVermelha(self, box_or_xyxy, annotated_frame):
        if hasattr(box_or_xyxy, "xyxy"):
            xyxy = _to_np(box_or_xyxy.xyxy[0]).tolist()
        else:
            xyxy = box_or_xyxy
        x1, y1, x2, y2 = map(int, xyxy[:4])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

# ---------------- Verificadores (mantidos originais) ----------------
class CotoveloAcimaCabeca:
    def __init__(self, keypoints):
        self.cotoveloDireito = keypoints[8]
        self.cotoveloEsquerdo = keypoints[7]
        self.nariz = keypoints[0]
    def calcular(self):
        return (self.cotoveloDireito[1] < self.nariz[1]) or (self.cotoveloEsquerdo[1] < self.nariz[1])

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

class inclinacaoFrontal:
    def __init__(self, keypoints):
        self.ombroDireito = keypoints[6]
        self.ombroEsquerdo = keypoints[5]
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.nariz = keypoints[0]
    def calcular(self):
        centroOmbrosY = (self.ombroDireito[1] + self.ombroEsquerdo[1]) / 2.0
        centroQuadrisY = (self.quadrilDireito[1] + self.quadrilEsquerdo[1]) / 2.0
        alturaCorpo = self.nariz[1] - centroQuadrisY
        if abs(alturaCorpo) < 1e-6:
            return 0.0
        return (centroOmbrosY - centroQuadrisY) / alturaCorpo

class AlinhamentoOmbros:
    def __init__(self, keypoints):
        self.ombro_direito = keypoints[6]
        self.ombro_esquerdo = keypoints[5]
    def verificar(self, limite_tolerancia=15):
        diferenca_altura = abs(self.ombro_direito[1] - self.ombro_esquerdo[1])
        return diferenca_altura > limite_tolerancia

class torcaoPescoco:
    def __init__(self, keypoints):
        self.nariz = keypoints[0]
        self.orelhaDi = keypoints[4]
        self.orelhaEs = keypoints[3]
        self.ombroEs = keypoints[5]
        self.ombroDi = keypoints[6]
    def calcular(self):
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

class InclinacaoLombar:
    def __init__(self, keypoints):
        self.ombro_esq = keypoints[5]
        self.ombro_dir = keypoints[6]
        self.quadril_esq = keypoints[11]
        self.quadril_dir = keypoints[12]
    def calcular(self):
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

class Agachamento:
    def __init__(self, keypoints):
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.joelhoDireito = keypoints[14]
        self.joelhoEsquerdo = keypoints[13]
    def calcular_angulo_joelho(self, lado):
        if lado == "direito":
            quadril, joelho = self.quadrilDireito, self.joelhoDireito
        else:
            quadril, joelho = self.quadrilEsquerdo, self.joelhoEsquerdo
        return abs(joelho[1] - quadril[1])