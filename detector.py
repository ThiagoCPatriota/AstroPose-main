# detector.py
import cv2
from ultralytics import YOLO
import torch
import time
import math
import numpy as np
import os
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

        # timers / estados
        self.tempo_cotovelo = {}
        self.tempo_inclinacao = {}
        self.tempo_inclinacao_frontal = {}
        self.tempoTorcao = {}
        self.tempoLombar = {}
        self.tempo_agachamento_incorreto = {}
        self.agachamento_iniciado = {}

        self._last_keypoints_ui = None
        self._last_issue_code = "OK"

        # Reconhecimento Facial
        self.face_recognizer = ReconhecimentoFacial()

        # Cadastro passo-a-passo
        self.cadastro_ativo = False
        self.nome_cadastro = None
        self.person_id = None
        self.n_shots = 0
        self.prompts = []
        self.shot_index = 0
        self.session_encodings = []
        self.session_paths = []
        self.session_person_dir = None
        self.session_safe_name = None
        self._duplicate_name_existing_ids = []

        # Mensagens rápidas (overlay visual)
        self._flash_msg = None
        self._flash_until = 0.0

        if not self.cap.isOpened():
            raise Exception("Erro ao acessar a webcam.")

    # ---------------- Flash overlay ----------------
    def flash_message(self, msg: str, seconds: float = 4.0):
        self._flash_msg = msg
        self._flash_until = time.time() + max(0.1, seconds)

    # ---------------- Cadastro passo-a-passo ----------------
    def iniciar_cadastro_astronauta(self, nome=None, n_shots=8):
        """Prepara sessão de cadastro; captura será feita por 'capturar_foto_cadastro()'."""
        self.cadastro_ativo = True
        self.nome_cadastro = (nome or "").strip()
        self.n_shots = max(1, int(n_shots))
        self.prompts = [
            "Frontal, rosto neutro",
            "Olhe levemente para ESQUERDA",
            "Olhe levemente para DIREITA",
            "Olhe um pouco PARA CIMA",
            "Olhe um pouco PARA BAIXO",
            "Incline a cabeça para ESQUERDA",
            "Incline a cabeça para DIREITA",
            "Frontal sorrindo (opcional)",
        ]
        if self.n_shots > len(self.prompts):
            self.prompts += ["Frontal extra"] * (self.n_shots - len(self.prompts))
        self.prompts = self.prompts[:self.n_shots]

        # status nome repetido?
        self._duplicate_name_existing_ids = self.face_recognizer.ids_by_name(self.nome_cadastro)

        # reset sessão
        self.shot_index = 0
        self.session_encodings = []
        self.session_paths = []

        # definir pasta com nome + ID
        safe = "_".join(self.nome_cadastro.split()) if self.nome_cadastro else f"astronauta_{int(time.time())}"
        self.person_id = self.face_recognizer.new_person_id()
        self.session_safe_name = f"{safe}__{self.person_id}"
        self.session_person_dir = os.path.join("astronautas", self.session_safe_name)
        os.makedirs(self.session_person_dir, exist_ok=True)

        return f"Modo cadastro ativo para: {self.nome_cadastro if self.nome_cadastro else 'novo astronauta'}"

    def capturar_foto_cadastro(self):
        """
        Captura UM frame, extrai encoding e salva a foto.
        Retorna dict:
          ok: bool, msg: str, step, total, remaining, prompt_next, done
        """
        if not self.cadastro_ativo:
            return {"ok": False, "msg": "Cadastro não está ativo."}
        if self.shot_index >= self.n_shots:
            return {"ok": False, "msg": "Todas as fotos já foram capturadas.", "done": True}

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return {"ok": False, "msg": "Falha ao capturar foto da câmera."}

        # localizar rosto e extrair encoding
        import face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        locs = face_recognition.face_locations(small, model="hog")
        if not locs:
            return {"ok": False, "msg": "Nenhum rosto detectado. Ajuste enquadramento/iluminação e tente novamente."}

        loc = max(locs, key=lambda ltbr: (ltbr[1]-ltbr[3])*(ltbr[2]-ltbr[0]))
        top, right, bottom, left = [v * 2 for v in loc]
        encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
        if not encs:
            return {"ok": False, "msg": "Não consegui extrair o rosto com qualidade. Tente novamente."}
        enc = encs[0]

        # salvar imagem
        idx1 = self.shot_index + 1
        filename = f"{self.session_safe_name}_{idx1:02d}.jpg"
        out_img = os.path.join(self.session_person_dir, filename)
        cv2.imwrite(out_img, frame)

        # acumular sessão
        self.session_encodings.append(enc)
        self.session_paths.append(out_img)
        self.shot_index += 1

        done = self.shot_index >= self.n_shots
        next_prompt = None if done else self.prompts[self.shot_index]
        return {
            "ok": True,
            "msg": f"Foto {idx1}/{self.n_shots} capturada.",
            "step": idx1,
            "total": self.n_shots,
            "remaining": max(0, self.n_shots - idx1),
            "prompt_next": next_prompt,
            "done": done
        }

    def concluir_cadastro_astronauta(self):
        """
        Finaliza sessão: cria contact sheet e grava encodings no pickle.
        Retorna dict {ok, msg, page_jpg, page_pdf, count, out_dir, id, name, duplicate, existing_ids}
        """
        if not self.cadastro_ativo:
            return {"ok": False, "msg": "Cadastro não está ativo."}
        if len(self.session_paths) == 0:
            self._reset_cadastro_session()
            return {"ok": False, "msg": "Nenhuma foto foi capturada."}

        # gerar contact sheet (título com Nome [ID])
        page_jpg, page_pdf = self.face_recognizer.make_contact_sheet(
            self.session_paths,
            self.session_person_dir,
            self.session_safe_name,
            title=f"Astronauta: {self.nome_cadastro} [{self.person_id}]"
        )

        # salvar encodings + IDs
        result = self.face_recognizer.finalize_person(
            self.nome_cadastro or self.session_safe_name,
            self.person_id,
            self.session_encodings,
            self.session_paths,
            overwrite=False
        )

        out_dir = self.session_person_dir
        count = len(self.session_paths)
        pid = self.person_id
        name = self.nome_cadastro

        duplicate = len(self._duplicate_name_existing_ids) > 0
        existing_ids = self._duplicate_name_existing_ids[:]

        # limpar estado
        self._reset_cadastro_session()

        out = {
            "ok": result.get("ok", False),
            "msg": result.get("msg", ""),
            "page_jpg": page_jpg,
            "page_pdf": page_pdf,
            "count": count,
            "out_dir": out_dir,
            "id": pid,
            "name": name,
            "duplicate": duplicate,
            "existing_ids": existing_ids
        }
        return out

    def _reset_cadastro_session(self):
        self.cadastro_ativo = False
        self.nome_cadastro = None
        self.person_id = None
        self.n_shots = 0
        self.prompts = []
        self.shot_index = 0
        self.session_encodings = []
        self.session_paths = []
        self.session_person_dir = None
        self.session_safe_name = None
        self._duplicate_name_existing_ids = []

    def esta_em_modo_cadastro(self):
        return self.cadastro_ativo

    # ---------------- Loop principal (pose+rosto) ----------------
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
        face_results = self.face_recognizer.reconhecer_rostos(frame, tolerance=0.6)
        for name, pid, (left, top, right, bottom), score in face_results:
            if score >= 0.75 and name != "Desconhecido":
                color = (0, 255, 0); label = f"{name} [{pid}] ({score:.0%})"
            elif score >= 0.5:
                color = (0, 255, 255); label = f"{name}? [{pid}] ({score:.0%})"
            else:
                color = (0, 0, 255); label = "Desconhecido"

            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(annotated_frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if score >= 0.75 and name != "Desconhecido":
                mensagens.append(f"Astronauta {name} [{pid}] detectado!")

        # Instruções visuais durante o cadastro passo-a-passo
        if self.cadastro_ativo:
            step = self.shot_index + 1
            total = self.n_shots
            prompt = self.prompts[self.shot_index] if self.shot_index < len(self.prompts) else "—"
            cv2.putText(annotated_frame, f"CADASTRO: passo {step}/{total}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Posição: {prompt}",
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Clique em CAPTURAR para tirar a foto.",
                        (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.nome_cadastro and self.person_id:
                cv2.putText(annotated_frame, f"Nome/ID: {self.nome_cadastro} [{self.person_id}]",
                            (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Feedback visual de "cadastrado!" (flash)
        if time.time() < self._flash_until and self._flash_msg:
            cv2.putText(annotated_frame, self._flash_msg,
                        (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 3)

        # ---------- Detecção de Pose ----------
        if res0.keypoints is None or res0.keypoints.xy is None:
            self._last_keypoints_ui = None
            self._last_issue_code = "OK"
            return mensagens, annotated_frame, None

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
                    val = inclinacaoFrontal(pessoaKey).calcular()
                    if val > 0.85:
                        inclinadoFrontal = True
                except Exception:
                    pass

                try:
                    desalinhado = AlinhamentoOmbros(pessoaKey).verificar()
                    if desalinhado:
                        mensagens.append(f"Pessoa {i + 1} com ombros desalinhados!")
                        self.bordaVermelha(box_tuple, annotated_frame)
                except Exception:
                    pass

                try:
                    agach = Agachamento(pessoaKey)
                    angulo_direito = agach.calcular_angulo_joelho("direito")
                    angulo_esquerdo = agach.calcular_angulo_joelho("esquerdo")
                    if angulo_direito and angulo_esquerdo and angulo_direito < 120 and angulo_esquerdo < 120:
                        self.agachamento_iniciado[i] = True
                    elif angulo_direito and angulo_esquerdo and angulo_direito >= 120 and angulo_esquerdo >= 120:
                        self.agachamento_iniciado[i] = False

                    if self.agachamento_iniciado.get(i, False):
                        if (angulo_direito and (angulo_direito < 45 or angulo_direito > 140)) or \
                           (angulo_esquerdo and (angulo_esquerdo < 45 or angulo_esquerdo > 140)):
                            agachamento_incorreto = True
                except Exception:
                    pass

                now = time.time()
                self._timer_check(self.tempo_cotovelo, i, now, 3.0,
                    lambda: (self.bordaVermelha(box_tuple, annotated_frame),
                             mensagens.append(f"Pessoa {i + 1} levantou o braço por mais de 3 segundos!")),
                    active=cotoveloAcima)

                self._timer_check(self.tempo_inclinacao, i, now, 3.0,
                    lambda: (self.bordaVermelha(box_tuple, annotated_frame),
                             mensagens.append(f"Pessoa {i + 1} está inclinada para trás por mais de 3 segundos!")),
                    active=inclinadoParaTras)

                self._timer_check(self.tempo_inclinacao_frontal, i, now, 3.0,
                    lambda: (self.bordaVermelha(box_tuple, annotated_frame),
                             mensagens.append(f"Pessoa {i + 1} está inclinada frontalmente por mais de 3 segundos!")),
                    active=inclinadoFrontal)

                self._timer_check(self.tempoLombar, i, now, 3.0,
                    lambda: (self.bordaVermelha(box_tuple, annotated_frame),
                             mensagens.append(f"Pessoa {i + 1} está com inclinação lombar excessiva por mais de 3 segundos!")),
                    active=inclinaLombar)

                self._timer_check(self.tempo_agachamento_incorreto, i, now, 3.0,
                    lambda: (self.bordaVermelha(box_tuple, annotated_frame),
                             mensagens.append(f"Pessoa {i + 1} está agachando incorretamente por mais de 3 segundos")),
                    active=agachamento_incorreto)

                try:
                    torceu = torcaoPescoco(pessoaKey).calcular()
                    self._timer_check(self.tempoTorcao, i, now, 3.0,
                        lambda: (self.bordaVermelha(box_tuple, annotated_frame),
                                 mensagens.append(f"Alerta! Pessoa {i + 1} torceu o pescoço por mais de 3 segundos!")),
                        active=torceu)
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

        return mensagens, annotated_frame, self._last_keypoints_ui

    def _timer_check(self, store, idx, now, delay_s, on_fire, active):
        if active:
            if idx not in store:
                store[idx] = now
            elif now - store[idx] >= delay_s:
                on_fire()
        else:
            store.pop(idx, None)

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

# ---------------- Verificadores ----------------
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
    """Ângulo do joelho (quadril-joelho-tornozelo)."""
    def __init__(self, keypoints):
        self.quadrilDireito   = keypoints[12]
        self.quadrilEsquerdo  = keypoints[11]
        self.joelhoDireito    = keypoints[14]
        self.joelhoEsquerdo   = keypoints[13]
        self.tornozeloDireito = keypoints[16]
        self.tornozeloEsquerdo= keypoints[15]

    @staticmethod
    def _angulo(v1, v2):
        v1 = np.array([v1[0], v1[1]], dtype=float)
        v2 = np.array([v2[0], v2[1]], dtype=float)
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def calcular_angulo_joelho(self, lado):
        if lado == "direito":
            Q, J, T = self.quadrilDireito, self.joelhoDireito, self.tornozeloDireito
        else:
            Q, J, T = self.quadrilEsquerdo, self.joelhoEsquerdo, self.tornozeloEsquerdo
        if Q is None or J is None or T is None:
            return None
        v1 = (Q[0]-J[0], Q[1]-J[1])  # joelho->quadril
        v2 = (T[0]-J[0], T[1]-J[1])  # joelho->tornozelo
        return self._angulo(v1, v2)
