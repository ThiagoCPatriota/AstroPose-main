import cv2
from ultralytics import YOLO
import torch
import time
import numpy as np

from src.analysis import Agachamento
from src.analysis import AlinhamentoOmbros
from src.analysis import CotoveloAcimaCabeca
from src.analysis import InclinacaoFrontal
from src.analysis import InclinacaoLombar
from src.analysis import InclinacaoParaTras
from src.analysis import TorcaoPescoco
from src.core.reconhecimento_facial import ReconhecimentoFacial


def _to_np(x):
    """Converte um tensor (CPU/CUDA) para um array numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def listar_resolucoes_suportadas(camera_index=0):
    """
    Verifica as resoluções comuns suportadas por uma câmera.

    Args:
        camera_index (int): O índice da câmera a ser testada.

    Returns:
        list: uma lista de tuplas (largura, altura) com as resoluções suportadas.
    """
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
    Classe principal para detecção de pose e análise de movimentos.

    Gerencia a captura da câmera, o modelo YOLO, o reconhecimento facial e
    a execução de várias análises de postura em tempo real.
    """
    def get_keypoints(self):
        """Retorna os últimos keypoints detectados para a pessoa principal."""
        return getattr(self, "_last_keypoints_ui", None)

    def get_issue_code(self):
        """Retorna o último código de problema de postura detectado."""
        return getattr(self, "_last_issue_code", "OK")

    def __init__(
        self,
        camera_index=0,
        model_path="models/yolov8n-pose.pt",
        width=640,
        height=480,
        frame_interval_ms=100,
        draw_annotations=False,
        force_cpu=False,
        face_recognition_enabled=True,
    ):
        """
        Inicializa o detector de pose.

        Args:
            camera_index (int): Índice da câmera a ser usada.
            model_path (str): Caminho para o arquivo do modelo YOLOv8-pose.
            width (int): Largura desejada para o frame da câmera.
            height (int): Altura desejada para o frame da câmera.
            frame_interval_ms (int): Intervalo em milissegundos entre as inferências.
            draw_annotations (bool): Se True, desenha o esqueleto completo no frame.
            force_cpu (bool): Se True, força o uso da CPU mesmo que uma GPU esteja disponível.
            face_recognition_enabled (bool): Se True, habilita o reconhecimento facial.
        """
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
        """Ativa o modo de cadastro de um novo astronauta."""
        self.cadastro_ativo = True
        self.nome_cadastro = nome

    def finalizar_cadastro_astronauta(self):
        """
        Finaliza o processo de cadastro, capturando o frame atual.

        Returns:
            tuple: Uma tupla (success, message) com o resultado do cadastro.
        """
        if self.cadastro_ativo and self.face_recognizer:
            try:
                # Captura o frame mais recente diretamente da câmera
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return False, "Falha ao capturar imagem da câmera."

                # Chama a função de cadastro (que agora não recebe mais o frame como argumento)
                success, message = self.face_recognizer.cadastrar_astronauta(frame, self.nome_cadastro)

                if success:
                    self.cadastro_ativo = False
                    self.nome_cadastro = None

                return success, message
            except Exception as e:
                print(f"Erro ao finalizar cadastro: {e}")
                self.cadastro_ativo = False
                self.nome_cadastro = None
                return False, f"Erro interno: {e}"
        return False, "O modo de cadastro não está ativo."

    def liberarRecursos(self):
        """Libera a câmera e fecha todas as janelas do OpenCV."""
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

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

    def detectar_pose(self):
        """
        Captura um frame da câmera, executa a detecção e todas as análises.

        Este é o método principal do loop de detecção. Ele gerencia o cache de
        resultados para otimizar o desempenho e aplica as análises de postura.

        Returns:
            tuple: Uma tupla contendo (mensagens, annotated_frame, keypoints).
                   - mensagens (list): Lista de alertas de postura.
                   - annotated_frame (np.array): Frame com as anotações visuais.
                   - keypoints (list): Lista de keypoints da pessoa principal.
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

        if(self.draw_annotations):
            annotated_frame = res0.plot()
        else:
            annotated_frame = frame.copy()
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
                    val = InclinacaoFrontal(pessoaKey).calcular()
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
                    torceu = TorcaoPescoco(pessoaKey).calcular()
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
