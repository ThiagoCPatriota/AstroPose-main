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

def listar_resolucoes_suportadas(camera_index=0):
    """Tenta aplicar resoluções comuns e retorna apenas as suportadas pela câmera"""
    common_resolutions = [
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1600, 1200),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ]

    cap = cv2.VideoCapture(camera_index)
    supported = []
    for w, h in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) == (w, h):
            supported.append((w, h))
    cap.release()
    return supported

class PoseDetector:
    """
    PoseDetector otimizado:
    - configurable: model_path, device (auto/force cpu), input resolution, frame interval (ms),
      draw_annotations (bool: desenhar esqueleto minimal), face recognition on/off.
    - caching de último resultado para frames pulados.
    """
    def get_keypoints(self):
        return getattr(self, "_last_keypoints_ui", None)

    def get_issue_code(self):
        return getattr(self, "_last_issue_code", "OK")

    def __init__(
        self,
        camera_index=0,
        model_path="yolov8n-pose.pt",
        width=640,
        height=480,
        frame_interval_ms=100,
        draw_annotations=False,
        force_cpu=False,
        face_recognition_enabled=True,
    ):
        self.camera_index = camera_index
        self.width = int(width)
        self.height = int(height)
        self.frame_interval_ms = int(frame_interval_ms)
        self.draw_annotations = bool(draw_annotations)
        self.face_recognition_enabled = bool(face_recognition_enabled)

        self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        # initialize model
        self.model_path = model_path
        # Load model (ultralytics)
        try:
            self.model = YOLO(self.model_path)
            if self.device == "cuda":
                try:
                    # put model on GPU if available
                    self.model.to("cuda:0")
                except Exception:
                    pass
        except Exception as e:
            raise Exception(f"Erro ao carregar modelo ({self.model_path}): {e}")

        # camera
        self.cap = cv2.VideoCapture(self.camera_index)
        # set requested capture resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # timers / caches
        self._last_infer_time = 0.0
        self._last_result_cache = None  # (mensagens, annotated_frame, keypoints)
        self._last_keypoints_ui = None
        self._last_issue_code = "OK"

        # temporizadores por pessoa (originais)
        self.tempo_cotovelo = {}
        self.tempo_inclinacao = {}
        self.tempo_inclinacao_frontal = {}
        self.tempoTorcao = {}
        self.tempoLombar = {}
        self.tempo_agachamento_incorreto = {}

        # Face recognizer (heavy) - keep but can be disabled
        self.face_recognizer = ReconhecimentoFacial() if self.face_recognition_enabled else None
        self.cadastro_ativo = False
        self.nome_cadastro = None

        if not self.cap.isOpened():
            raise Exception("Erro ao acessar a webcam.")

    def iniciar_cadastro_astronauta(self, nome=None):
        self.cadastro_ativo = True
        self.nome_cadastro = nome

    def finalizar_cadastro_astronauta(self, frame):
        if getattr(self, "cadastro_ativo", False):
            if self.face_recognizer:
                try:
                    success = self.face_recognizer.cadastrar_astronauta(frame, self.nome_cadastro)
                    self.cadastro_ativo = False
                    self.nome_cadastro = None
                    return success
                except Exception as e:
                    print(f"Erro ao cadastrar astronauta: {e}")
                    self.cadastro_ativo = False
                    self.nome_cadastro = None
                    return False
        return False

    def set_resolution(self, width, height):
        self.width = int(width)
        self.height = int(height)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            pass

    def set_frame_interval(self, ms):
        self.frame_interval_ms = int(ms)

    def liberarRecursos(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # ------------------------------
    # Funções utilitárias (mantidas/otimizadas)
    # ------------------------------
    def bordaVermelha(self, box_or_xyxy, annotated_frame):
        if hasattr(box_or_xyxy, "xyxy"):
            xyxy = _to_np(box_or_xyxy.xyxy[0]).tolist()
        else:
            xyxy = box_or_xyxy
        x1, y1, x2, y2 = map(int, xyxy[:4])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # ------------------------------
    # Método principal de detecção otimizado
    # ------------------------------
    def detectar_pose(self):
        """
        Retorna (mensagens, annotated_frame, keypoints)
        - annotated_frame: BGR image pronta para exibição
        - keypoints: lista de keypoints (x,y,conf) para a pessoa principal (ou None)
        Implementa:
        - pular inferências por intervalo (frame_interval_ms)
        - redimensionamento controlado
        - evitar res0.plot() por padrão (menos custo)
        - escala de keypoints do tamanho de entrada de volta para frame original
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise Exception("Erro ao capturar o frame da webcam.")

        now = time.time()
        elapsed_ms = (now - self._last_infer_time) * 1000.0
        # se ainda não atingiu o intervalo, retorna dado cacheado (se houver)
        if self._last_result_cache is not None and elapsed_ms < self.frame_interval_ms:
            # atualizar overlay leve (por ex: FPS/tempo) não é necessário - retornamos cache
            return self._last_result_cache

        # Preprocess: se for diferente da resolução solicitada, redimensiona para entrada do modelo
        original_h, original_w = frame.shape[:2]
        input_w, input_h = self.width, self.height
        need_resize = (original_w != input_w) or (original_h != input_h)
        frame_for_model = cv2.resize(frame, (input_w, input_h)) if need_resize else frame

        # inference (torch.no_grad já dentro do modelo, mas mantemos)
        with torch.no_grad():
            # passamos o frame (BGR) direto - ultralytics aceita numpy arrays
            try:
                results = self.model(frame_for_model, verbose=False)
            except Exception as e:
                # falha na inferência: retorna frame cru
                annotated_frame = frame.copy()
                self._last_result_cache = ([], annotated_frame, None)
                self._last_infer_time = time.time()
                return self._last_result_cache

        res0 = results[0]

        # parse detections sem usar res0.plot() (muito mais rápido)
        annotated_frame = frame.copy()  # desenharemos apenas o necessário
        mensagens = []

        # ---------- Reconhecimento Facial (mantido, mas só se ativado) ----------
        if self.face_recognizer:
            try:
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
                    cv2.putText(annotated_frame, label, (left, max(15, top - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if confidence > 0.7 and name != "Desconhecido":
                        mensagens.append(f"Astronauta {name} detectado!")
            except Exception:
                # não bloquear pipeline por erros em reconhecimento facial
                pass

        # ---------- Pose (boxes/keypoints) ----------
        n = len(res0.boxes) if res0.boxes is not None else 0
        if n > 0:
            # obter keypoints e confs (coordenadas relativas à imagem de entrada)
            try:
                xy_all = _to_np(res0.keypoints.xy)  # shape (n, 17, 2)
                conf_all = _to_np(res0.keypoints.conf)
            except Exception:
                # fallback
                xy_all = np.zeros((n, 17, 2), dtype=np.float32)
                conf_all = np.ones((n, 17), dtype=np.float32)

            xyxy_all = _to_np(res0.boxes.xyxy)  # boxes no espaço de entrada do modelo

            # escolher a maior pessoa (maior área)
            areas = (xyxy_all[:, 2] - xyxy_all[:, 0]) * (xyxy_all[:, 3] - xyxy_all[:, 1])
            idx_max = int(np.argmax(areas))

            # Escalar keypoints para dimensão original da câmera
            scale_x = original_w / float(input_w)
            scale_y = original_h / float(input_h)

            # formar keypoints no espaço original
            kp_scaled_all = []
            for person_kps, person_confs in zip(xy_all, conf_all):
                kp_person = []
                for (x, y), c in zip(person_kps, person_confs):
                    kp_person.append((float(x) * scale_x, float(y) * scale_y, float(c)))
                kp_scaled_all.append(kp_person)

            # escolher keypoints da pessoa principal
            kp0 = [(float(x), float(y), float(c)) for (x, y, c) in kp_scaled_all[idx_max]]
            self._last_keypoints_ui = kp0

            # desenhar sobre annotated_frame somente se draw_annotations True (opcional)
            if self.draw_annotations:
                # desenhar caixa maior e pontos básicos para cada pessoa (apenas para economia)
                for i in range(n):
                    box = xyxy_all[i]
                    x1, y1, x2, y2 = int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # desenhar pontos da pessoa i (somente pontos com confiança razoável)
                    for (x, y), c in zip(xy_all[i], conf_all[i]):
                        if c > 0.3:
                            cv2.circle(annotated_frame, (int(x * scale_x), int(y * scale_y)), 3, (255, 255, 255), -1)

            # Agora executamos as verificações originais de postura para cada pessoa (otimizando try/except)
            for i in range(n):
                pessoaKey = kp_scaled_all[i]
                xyxy = (xyxy_all[i][0] * scale_x,
                        xyxy_all[i][1] * scale_y,
                        xyxy_all[i][2] * scale_x,
                        xyxy_all[i][3] * scale_y)

                cotoveloAcima = False
                inclinadoParaTras = False
                inclinadoFrontal = False
                torceu = False
                inclinaLombar = False
                agachamento_incorreto = False
                agachamento_iniciado = {}
                desalinhado = False

                # Checks originais (mantidos) - mantemos proteções individuais
                try:
                    if CotoveloAcimaCabeca(pessoaKey).calcular():
                        cotoveloAcima = True
                except Exception:
                    pass

                try:
                    if InclinacaoLombar(pessoaKey).calcular():
                        inclinaLombar = True
                except Exception:
                    pass

                try:
                    if InclinacaoParaTras(pessoaKey).calcular():
                        inclinadoParaTras = True
                except Exception:
                    pass

                try:
                    val = inclinacaoFrontal(pessoaKey).calcular()
                    if val > 0.85:
                        inclinadoFrontal = True
                except Exception:
                    pass

                try:
                    desalinhado = AlinhamentoOmbros(pessoaKey).verificar()
                    if desalinhado:
                        mensagens.append(f"Pessoa {i + 1} com ombros desalinhados!")
                        self.bordaVermelha(xyxy, annotated_frame)
                except Exception:
                    pass

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
                except Exception:
                    pass

                # Temporizações originais (mantidas)
                if cotoveloAcima:
                    if i not in self.tempo_cotovelo:
                        self.tempo_cotovelo[i] = time.time()
                    elif time.time() - self.tempo_cotovelo[i] >= 3:
                        self.bordaVermelha(xyxy, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} levantou o braço por mais de 3 segundos!")
                else:
                    self.tempo_cotovelo.pop(i, None)

                if inclinadoParaTras:
                    if i not in self.tempo_inclinacao:
                        self.tempo_inclinacao[i] = time.time()
                    elif time.time() - self.tempo_inclinacao[i] >= 3:
                        self.bordaVermelha(xyxy, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está inclinada para trás por mais de 3 segundos!")
                else:
                    self.tempo_inclinacao.pop(i, None)

                if inclinadoFrontal:
                    if i not in self.tempo_inclinacao_frontal:
                        self.tempo_inclinacao_frontal[i] = time.time()
                    elif time.time() - self.tempo_inclinacao_frontal[i] >= 3:
                        self.bordaVermelha(xyxy, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está inclinada frontalmente por mais de 3 segundos!")
                else:
                    self.tempo_inclinacao_frontal.pop(i, None)

                if inclinaLombar:
                    if i not in self.tempoLombar:
                        self.tempoLombar[i] = time.time()
                    elif time.time() - self.tempoLombar[i] >= 3:
                        self.bordaVermelha(xyxy, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está com inclinação lombar excessiva por mais de 3 segundos!")
                else:
                    self.tempoLombar.pop(i, None)

                if agachamento_incorreto:
                    if i not in self.tempo_agachamento_incorreto:
                        self.tempo_agachamento_incorreto[i] = time.time()
                    elif time.time() - self.tempo_agachamento_incorreto[i] >= 3:
                        self.bordaVermelha(xyxy, annotated_frame)
                        mensagens.append(f"Pessoa {i + 1} está agachando incorretamente por mais de 3 segundos")
                else:
                    self.tempo_agachamento_incorreto.pop(i, None)

                try:
                    torceu = torcaoPescoco(pessoaKey).calcular()
                    if torceu:
                        if i not in self.tempoTorcao:
                            self.tempoTorcao[i] = time.time()
                        elif time.time() - self.tempoTorcao[i] >= 3:
                            self.bordaVermelha(xyxy, annotated_frame)
                            mensagens.append(f"Alerta! Pessoa {i + 1} torceu o pescoço por mais de 3 segundos!")
                    else:
                        self.tempoTorcao.pop(i, None)
                except Exception:
                    pass

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

        if getattr(self, "cadastro_ativo", False):
            cv2.putText(
                annotated_frame,
                "MODO CADASTRO - Posicione o rosto",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),  # amarelo
                2
            )

        # cache result and atualizar timer
        result = (mensagens, annotated_frame, self._last_keypoints_ui)
        self._last_result_cache = result
        self._last_infer_time = time.time()
        return result


# ---------------- Verificadores (mantidos/otimizados) ----------------
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
