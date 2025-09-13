import os
import math
import time

from PySide6.QtCore import (Qt, QThread, QObject, Signal, Slot)
from PySide6.QtGui import (QPainter, QColor, QFont, QImage, QPixmap)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QStackedWidget, QLineEdit, QGroupBox, QFormLayout,
    QStatusBar, QButtonGroup, QComboBox, QSpinBox, QCheckBox, QMessageBox
)
from PIL import Image
import cv2

from src.core.detector import PoseDetector
from src.utils import _compute_angles, _choose_sprite_auto, _normalize_keypoints, KP, listar_resolucoes_suportadas

# ---------------- constants / maps ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_DIR = os.path.join(BASE_DIR, "../../assets/img")
SPRITE_MAP = {
    "Neutro":     os.path.join(SPRITE_DIR, "neutral.png"),
    "Joinha":     os.path.join(SPRITE_DIR, "thumbsup.png"),
    "Apontando":  os.path.join(SPRITE_DIR, "point.png"),
    "T-pose":     os.path.join(SPRITE_DIR, "tpose.png"),
}
SPRITE_CHOICES = ["Auto", "Neutro", "Joinha", "Apontando", "T-pose"]

# ---------------- CoachCanvas (QWidget) ----------------
class CoachCanvas(QWidget):
    """
    Widget customizado para desenhar o astronauta-alvo e o esqueleto do utilizador.
    """
    def __init__(self, colors=None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 320)
        self.colors = colors or {
            "card": "#0b1220", "muted": "#94a3b8", "danger": "#f87171"
        }
        # state
        self.last_current_kp = None
        self.target_angles = None
        self.target_label = None
        self.target_raw_kp = None
        self.sprite_choice = "Auto"
        self.sprite_path = None
        self.sprite_pix = None
        self.external_issue = None
        self.tolerance_ok = 14
        self.tolerance_warn = 28

    def set_sprite_choice(self, choice):
        self.sprite_choice = choice
        if choice in SPRITE_MAP and os.path.exists(SPRITE_MAP[choice]):
            self.sprite_path = SPRITE_MAP[choice]
            self._load_sprite()
        elif choice != "Auto":
            self.sprite_path = None
            self.sprite_pix = None
        self.update()

    def set_external_issue(self, issue):
        self.external_issue = issue

    def set_target_from_current(self, raw_kp):
        if not raw_kp:
            return
        self.target_raw_kp = raw_kp
        self.target_angles = _compute_angles(raw_kp)
        self.target_label = _choose_sprite_auto(raw_kp) or "Neutro"
        if self.sprite_choice == "Auto":
            p = SPRITE_MAP.get(self.target_label)
            self.sprite_path = p if p and os.path.exists(p) else None
            self._load_sprite()
        self.update()

    def clear_target(self):
        self.target_angles = None
        self.target_label = None
        self.target_raw_kp = None
        if self.sprite_choice == "Auto":
            self.sprite_path = None
            self.sprite_pix = None
        self.update()

    def update_pose(self, raw_kp):
        self.last_current_kp = raw_kp
        if self.sprite_choice == "Auto":
            self._choose_sprite_auto_corrective()
        self.update()

    def _issue_to_sprite(self):
        if not self.external_issue or self.external_issue == "OK":
            return None
        mapping = {
            "BACK_TILT": "Neutro",
            "FRONT_TILT": "Neutro",
            "LUMBAR": "Neutro",
            "ELBOW_UP": "Joinha",
            "SHOULDER_MISALIGN": "T-pose",
            "NECK_TWIST": "Apontando",
        }
        return mapping.get(self.external_issue)

    def _choose_sprite_auto_corrective(self):
        name_ext = self._issue_to_sprite()
        if name_ext and self.sprite_choice == "Auto":
            p = SPRITE_MAP.get(name_ext)
            if p and os.path.exists(p):
                self.sprite_path = p
                self._load_sprite()
                return

        if self.target_angles is None or self.last_current_kp is None:
            name = _choose_sprite_auto(self.last_current_kp or [])
            p = SPRITE_MAP.get(name)
            self.sprite_path = p if p and os.path.exists(p) else None
            self._load_sprite()
            return

        curr = _compute_angles(self.last_current_kp)
        diffs = {}
        for j, tgt in self.target_angles.items():
            a = curr.get(j)
            if tgt is None or a is None:
                continue
            diffs[j] = abs(a - tgt)

        if not diffs:
            name = self.target_label or "Neutro"
            p = SPRITE_MAP.get(name)
            self.sprite_path = p if p and os.path.exists(p) else None
            self._load_sprite()
            return

        worst_joint = max(diffs, key=diffs.get)
        worst_val = diffs[worst_joint]

        if worst_joint == "trunk_tilt" and worst_val >= max(self.tolerance_warn, 20):
            name = "Neutro"
        elif worst_joint in ("l_elbow", "r_elbow"):
            tgt = self.target_angles.get(worst_joint)
            name = "Joinha" if (tgt is not None and tgt < 120) else "Apontando"
        elif worst_joint in ("l_shoulder", "r_shoulder"):
            l_tgt, r_tgt = self.target_angles.get("l_shoulder"), self.target_angles.get("r_shoulder")
            if (l_tgt and 70 <= l_tgt <= 110) and (r_tgt and 70 <= r_tgt <= 110):
                name = "T-pose"
            else:
                name = self.target_label or "Neutro"
        else:
            name = self.target_label or "Neutro"

        p = SPRITE_MAP.get(name)
        self.sprite_path = p if p and os.path.exists(p) else None
        self._load_sprite()

    def _load_sprite(self):
        try:
            if self.sprite_path and os.path.exists(self.sprite_path):
                img = Image.open(self.sprite_path).convert("RGBA")
                w, h = img.size
                max_w, max_h = 200, 220
                scale = min(max_w / float(w), max_h / float(h), 1.0)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                data = img.tobytes("raw", "RGBA")
                qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
                self.sprite_pix = QPixmap.fromImage(qimg)
            else:
                self.sprite_pix = None
        except Exception:
            self.sprite_pix = None

    def paintEvent(self, ev, BONES=None):
        p = QPainter(self)
        rect = self.rect()
        # background
        p.fillRect(rect, QColor(self.colors.get("card", "#0b1220")))
        # Header labels
        p.setPen(QColor(self.colors.get("muted", "#94a3b8")))
        f = QFont("Segoe UI", 9, QFont.Bold)
        p.setFont(f)
        p.drawText(rect.width() * 0.12, 20, "POSE-ALVO")
        p.drawText(rect.width() * 0.62, 20, "VOCÊ")

        # draw sprite target (left)
        if self.sprite_pix:
            tx = int(rect.width() * 0.12)
            ty = int(rect.height() * 0.12)
            p.drawPixmap(tx, ty, self.sprite_pix)

        # draw skeleton for current pose (right side)
        if not self.last_current_kp:
            f2 = QFont("Segoe UI", 11)
            p.setFont(f2)
            p.setPen(QColor(self.colors.get("muted", "#94a3b8")))
            p.drawText(rect, Qt.AlignCenter, "Aguardando pose…")
            p.end()
            return

        # normalize keypoints to widget coordinates
        kp_norm = _normalize_keypoints(self.last_current_kp, rect.width(), rect.height(), side="right")
        if not kp_norm:
            p.end()
            return

        # determine highlight bones based on target angles
        highlight_bones = set()
        if self.target_angles:
            curr = _compute_angles(self.last_current_kp)
            diffs = {}
            for joint, a_tgt in self.target_angles.items():
                a_cur = curr.get(joint)
                if a_tgt is None or a_cur is None:
                    continue
                diff = abs(a_cur - a_tgt)
                diffs[joint] = diff
                if diff > self.tolerance_warn:
                    if joint == "l_elbow":
                        highlight_bones.update({(KP["l_shoulder"], KP["l_elbow"]), (KP["l_elbow"], KP["l_wrist"])})
                    elif joint == "r_elbow":
                        highlight_bones.update({(KP["r_shoulder"], KP["r_elbow"]), (KP["r_elbow"], KP["r_wrist"])})
                    elif joint == "l_knee":
                        highlight_bones.update({(KP["l_hip"], KP["l_knee"]), (KP["l_knee"], KP["l_ankle"])})
                    elif joint == "r_knee":
                        highlight_bones.update({(KP["r_hip"], KP["r_knee"]), (KP["r_knee"], KP["r_ankle"])})
                    elif joint in ("l_shoulder", "r_shoulder", "trunk_tilt"):
                        highlight_bones.update({(KP["l_shoulder"], KP["r_hip"]), (KP["r_shoulder"], KP["l_hip"]), (KP["l_shoulder"], KP["r_shoulder"])})

        # draw bones
        for (i, j) in BONES:
            pi = kp_norm[i] if i < len(kp_norm) else None
            pj = kp_norm[j] if j < len(kp_norm) else None
            if not pi or not pj:
                continue
            # if too large skip
            if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) > 800:
                continue
            col = QColor(self.colors.get("danger")) if ((i, j) in highlight_bones or (j, i) in highlight_bones) else QColor("#e5e7eb")
            pen = p.pen()
            pen.setWidth(3)
            pen.setColor(col)
            p.setPen(pen)
            p.drawLine(int(pi[0]), int(pi[1]), int(pj[0]), int(pj[1]))

        # draw joints
        for pnt in kp_norm:
            if not pnt:
                continue
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#e5e7eb"))
            p.drawEllipse(int(pnt[0]) - 4, int(pnt[1]) - 4, 8, 8)

        # bottom conformity text if target exists
        if self.target_angles:
            curr = _compute_angles(self.last_current_kp)
            conf_ok = 0; total = 0
            diffs = {}
            for joint, a_tgt in self.target_angles.items():
                a_cur = curr.get(joint)
                if a_tgt is None or a_cur is None:
                    continue
                total += 1
                diff = abs(a_cur - a_tgt)
                diffs[joint] = diff
                if diff <= self.tolerance_ok:
                    conf_ok += 1
            if total > 0:
                conf = int(100 * (conf_ok / max(1, total)))
                bottom_text = f"Conformidade: {conf}%"
                # top3 diffs
                items = sorted(diffs.items(), key=lambda kv: -kv[1])[:3]
                for j, d in items:
                    bottom_text += f"  •  {j.replace('_',' ').title()}: {d:.0f}°"
                f3 = QFont("Segoe UI", 9)
                p.setFont(f3)
                p.setPen(QColor(self.colors.get("muted", "#94a3b8")))
                p.drawText(rect.width()/2 - 140, rect.height() - 12, bottom_text)

        p.end()

# ==========================
# Detection worker (QThread + QObject)
# ==========================
class DetectionWorker(QObject):
    """
    Worker que executa a detecção em uma thread separada para não travar a UI.
    """
    frame_ready = Signal(object)          # numpy RGB frame
    annotated_ready = Signal(object)      # numpy BGR annotated (para fallback)
    messages_ready = Signal(list)
    keypoints_ready = Signal(object)
    issue_ready = Signal(object)
    finished = Signal()
    fps_signal = Signal(float)
    res_signal = Signal(str)
    registration_finished = Signal(bool, str)

    def __init__(self, config=None):
        super().__init__()
        self._running = False
        self.pose_detector = None
        self.config = config or {}

    @Slot()
    def process_registration_capture(self):
        if self.pose_detector:
            success, message = self.pose_detector.finalizar_cadastro_astronauta()
            self.registration_finished.emit(success, message)

    @Slot()
    def start_worker(self):
        try:
            # criar detector com configurações fornecidas
            self.pose_detector = PoseDetector(
                camera_index=self.config.get("camera_index", 0),
                model_path=self.config.get("model_path", "models/yolov8n-pose.pt"),
                width=self.config.get("width", 640),
                height=self.config.get("height", 480),
                frame_interval_ms=self.config.get("frame_interval_ms", 100),
                draw_annotations=self.config.get("draw_annotations", False),
                force_cpu=self.config.get("force_cpu", False),
                face_recognition_enabled=self.config.get("face_recognition_enabled", True),
            )
        except Exception as e:
            self.messages_ready.emit([f"Erro ao inicializar detector: {e}"])
            self.finished.emit()
            return

        self._running = True
        prev_time = time.time()
        while self._running:
            try:
                result = self.pose_detector.detectar_pose()
                got_kp = False
                if isinstance(result, tuple) and len(result) == 3:
                    mensagens, annotated_frame, keypoints = result
                    got_kp = True
                else:
                    mensagens, annotated_frame = result
                    keypoints = None

                if mensagens:
                    self.messages_ready.emit(mensagens)
                else:
                    self.messages_ready.emit([])

                # annotated_frame may be BGR; convert to RGB for display
                if annotated_frame is not None:
                    try:
                        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(annotated_rgb)
                        self.annotated_ready.emit(annotated_frame)
                        h, w, _ = annotated_rgb.shape
                        self.res_signal.emit(f"{w}x{h}")
                    except Exception:
                        self.frame_ready.emit(annotated_frame)
                        self.annotated_ready.emit(annotated_frame)

                if got_kp and keypoints is not None:
                    self.keypoints_ready.emit(keypoints)
                else:
                    try:
                        kp = self.pose_detector.get_keypoints()
                        if kp:
                            self.keypoints_ready.emit(kp)
                    except Exception:
                        pass

                # issue code
                try:
                    if hasattr(self.pose_detector, "get_issue_code"):
                        self.issue_ready.emit(self.pose_detector.get_issue_code())
                except Exception:
                    pass

                # fps
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    self.fps_signal.emit(1.0 / dt)

                time.sleep(0.01)
            except Exception as e:
                self.messages_ready.emit([f"Erro interno na detecção: {e}"])
                break

        # cleanup
        try:
            if self.pose_detector:
                self.pose_detector.liberarRecursos()
        except Exception:
            pass
        self.finished.emit()

    def stop(self):
        self._running = False

# ==========================
# Main Window
# ==========================
class AstroPoseMainWindow(QMainWindow):
    """
    Janela principal da aplicação AstroPose.

    Gerencia a interface do usuário (UI), inicia e para a detecção, e exibe
    os resultados das análises de postura.
    """
    def __init__(self):
        """Inicializa a janela principal e configura a UI."""
        super().__init__()
        self.setWindowTitle("AstroPose — Coach de Postura (Demo)")
        self.resize(1200, 760)
        self.pose_detector = None
        self.worker_thread = None
        self.worker = None
        self.COLORS = {
            "card": "#0b1220", "muted": "#94a3b8", "danger": "#f87171"
        }
        self.cadastro_ativo = False
        self._last_keypoints = None

        self._setup_ui()
        self._popular_resolucoes()
        self.combo_res.currentIndexChanged.connect(self._on_resolution_change)

    def _popular_resolucoes(self):
        """Preenche o ComboBox com as resoluções suportadas pela câmara."""
        resolucoes = listar_resolucoes_suportadas(0)  # índice da webcam
        self.combo_res.clear()
        for w, h in resolucoes:
            self.combo_res.addItem(f"{w}x{h}", (w, h))

    def _on_resolution_change(self, index):
        """
        Slot chamado quando o utilizador muda a resolução no ComboBox.
        Aplica a nova resolução na câmara se a deteção estiver ativa.
        """
        if index >= 0:
            w, h = self.combo_res.itemData(index)
            if self.worker and self.worker.pose_detector:
                # Atualiza atributos
                self.worker.pose_detector.width = w
                self.worker.pose_detector.height = h
                # Aplica na câmera
                cap = self.worker.pose_detector.cap
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

                # Confirmação
                atual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                atual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.info_text.append(
                    f"Resolução solicitada {w}x{h}, em uso {atual_w}x{atual_h}"
                )

    def _setup_ui(self):
        """Constrói todos os widgets e layouts da interface gráfica."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left: Camera
        cam_group = QGroupBox("Visualização da Câmera")
        cam_group.setStyleSheet(self._card_style())
        cam_layout = QVBoxLayout(cam_group)
        cam_layout.setContentsMargins(10, 12, 10, 10)
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(720, 480)
        self.camera_label.setStyleSheet("background-color: #000000; border-radius:6px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        cam_layout.addWidget(self.camera_label)

        # ----------------- CONFIG CONTROLS -----------------
        config_row = QHBoxLayout()
        config_row.setSpacing(8)

        self.combo_res = QComboBox()
        config_row.addWidget(self.combo_res)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(10, 2000)
        self.spin_interval.setValue(100)
        self.spin_interval.setSuffix(" ms")
        config_row.addWidget(self.spin_interval)

        self.chk_annotations = QCheckBox("Desenhar anotações")
        self.chk_annotations.setChecked(False)
        config_row.addWidget(self.chk_annotations)

        self.chk_force_cpu = QCheckBox("Forçar CPU")
        self.chk_force_cpu.setChecked(False)
        config_row.addWidget(self.chk_force_cpu)

        self.input_model = QLineEdit("yolov8n-pose.pt")
        self.input_model.setMaximumWidth(180)
        config_row.addWidget(self.input_model)

        cam_layout.addLayout(config_row)

        cam_controls = QHBoxLayout()
        cam_controls.setSpacing(8)
        self.btn_start = QPushButton("Iniciar Detecção")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.run_detection)
        cam_controls.addWidget(self.btn_start)

        self.status_small = QLabel("Aguardando início")
        self.status_small.setStyleSheet("color: #9aa8bf;")
        cam_controls.addWidget(self.status_small, alignment=Qt.AlignRight)
        cam_layout.addLayout(cam_controls)

        main_layout.addWidget(cam_group, stretch=2)

        # Right: segmented + stacked
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        seg_bar = QHBoxLayout()
        seg_bar.setSpacing(6)
        # btn_group será criado por _make_segment_button caso não exista
        self.btn_resumo = self._make_segment_button("Resumo", checked=True)
        self.btn_coach = self._make_segment_button("Coach")
        self.btn_metricas = self._make_segment_button("Métricas")
        self.btn_astronautas = self._make_segment_button("Astronautas")

        seg_bar.addWidget(self.btn_resumo)
        seg_bar.addWidget(self.btn_coach)
        seg_bar.addWidget(self.btn_metricas)
        seg_bar.addWidget(self.btn_astronautas)
        right_layout.addLayout(seg_bar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_resumo())
        self.stack.addWidget(self._page_coach())
        self.stack.addWidget(self._page_metricas())
        self.stack.addWidget(self._page_astronautas())
        right_layout.addWidget(self.stack, stretch=1)

        footer_box = QGroupBox()
        footer_box.setStyleSheet(self._card_style())
        footer_layout = QVBoxLayout(footer_box)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        self.btn_close = QPushButton("Fechar")
        self.btn_close.clicked.connect(self.close)
        self.btn_close.setFixedHeight(40)
        footer_layout.addWidget(self.btn_close)
        right_layout.addWidget(footer_box)

        main_layout.addWidget(right_panel, stretch=1)

        # connect segmented buttons
        self.btn_resumo.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_coach.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_metricas.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        self.btn_astronautas.clicked.connect(lambda: self.stack.setCurrentIndex(3))

        # status bar
        status = QStatusBar()
        self.setStatusBar(status)
        self.lbl_fps = QLabel("FPS: —")
        self.lbl_res = QLabel("Res: —")
        self.lbl_mode = QLabel("Modo: Terra↔Microg (visual)")
        status.addPermanentWidget(self.lbl_fps, 1)
        status.addPermanentWidget(self.lbl_res, 1)
        status.addPermanentWidget(self.lbl_mode, 1)

    # ----------------- pages -----------------
    def _page_resumo(self):
        """Cria o widget da página 'Resumo'."""
        page = QWidget()
        l = QVBoxLayout(page)
        l.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Mensagens / Ações Detectadas")
        box.setStyleSheet(self._card_style())
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 10, 8, 8)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Informações: \nNenhuma ação detectada.")
        box_layout.addWidget(self.info_text)

        row = QHBoxLayout()
        self.btn_set_target = QPushButton("Definir pose-alvo (agora)")
        self.btn_set_target.clicked.connect(self._set_target_now)
        self.btn_clear_target = QPushButton("Limpar pose-alvo")
        self.btn_clear_target.clicked.connect(self._clear_target)
        row.addWidget(self.btn_set_target)
        row.addWidget(self.btn_clear_target)
        box_layout.addLayout(row)

        l.addWidget(box)
        return page

    def _page_coach(self):
        """Cria o widget da página 'Coach'."""
        page = QWidget()
        l = QVBoxLayout(page)
        l.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Coach — Pose-alvo vs Você")
        box.setStyleSheet(self._card_style())
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 10, 8, 8)

        self.coach_canvas = CoachCanvas(colors=self.COLORS)
        box_layout.addWidget(self.coach_canvas)

        btns = QHBoxLayout()
        self.btn_set_target_coach = QPushButton("Definir Pose-Alvo")
        self.btn_set_target_coach.clicked.connect(self._set_target_now)
        self.btn_clear_target_coach = QPushButton("Limpar Pose")
        self.btn_clear_target_coach.clicked.connect(self._clear_target)
        btns.addWidget(self.btn_set_target_coach)
        btns.addWidget(self.btn_clear_target_coach)
        box_layout.addLayout(btns)

        l.addWidget(box)
        return page

    def _page_metricas(self):
        """Cria o widget da página 'Métricas'."""
        page = QWidget()
        l = QVBoxLayout(page)
        l.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Indicadores (MVP)")
        box.setStyleSheet(self._card_style())
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(8, 10, 8, 8)

        self.metric_conform = QLabel("Conformidade de postura: —")
        self.metric_latency = QLabel("Latência média (estimada): —")
        self.metric_epi = QLabel("Alertas EPI: —")
        for w in (self.metric_conform, self.metric_latency, self.metric_epi):
            w.setStyleSheet("color: #dbe7ff;")
            box_layout.addWidget(w)
        box_layout.addStretch(1)

        note = QLabel("*Defina uma pose-alvo para ativar a comparação articulada.\n*Ângulos: ombros, cotovelos, joelhos e inclinação do tronco.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa8bf; font-size: 11px;")
        box_layout.addWidget(note)

        l.addWidget(box)
        return page

    def _page_astronautas(self):
        """Cria o widget da página 'Astronautas' para gestão de cadastros."""
        page = QWidget()
        l = QVBoxLayout(page)
        l.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Gerenciar Astronautas")
        box.setStyleSheet(self._card_style())
        box_layout = QFormLayout(box)
        box_layout.setContentsMargins(8, 10, 8, 8)
        self.input_name = QLineEdit()
        box_layout.addRow("Nome:", self.input_name)

        btn_row = QHBoxLayout()
        self.btn_register = QPushButton("Cadastrar")
        self.btn_register.clicked.connect(self._iniciar_cadastro)
        self.btn_capture = QPushButton("Capturar (usar câmera)")
        self.btn_capture.clicked.connect(self._capturar_cadastro)
        btn_row.addWidget(self.btn_register)
        btn_row.addWidget(self.btn_capture)
        box_layout.addRow(btn_row)

        l.addWidget(box)
        return page

    # ----------------- Helpers -----------------
    def _make_segment_button(self, text, checked=False):
        """
        Cria um botão estilizado para a barra de navegação entre páginas.

        Args:
            text (str): O texto do botão.
            checked (bool): Se o botão deve iniciar como selecionado.

        Returns:
            QPushButton: O widget do botão criado.
        """
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(36)
        btn.setStyleSheet(self._segment_button_style())
        # Garantir que btn_group exista (cria se necessário)
        try:
            bg = getattr(self, "btn_group", None)
            if bg is None:
                self.btn_group = QButtonGroup(self)
                self.btn_group.setExclusive(True)
            self.btn_group.addButton(btn)
        except Exception:
            pass
        return btn

    def _gather_settings(self):
        """
        Reúne todas as configurações selecionadas na UI para iniciar o detector.

        Returns:
            dict: Um dicionário com todas as configurações.
        """
        res_text = self.combo_res.currentText()
        w, h = map(int, res_text.split("x"))
        cfg = {
            "width": w,
            "height": h,
            "frame_interval_ms": int(self.spin_interval.value()),
            "draw_annotations": bool(self.chk_annotations.isChecked()),
            "force_cpu": bool(self.chk_force_cpu.isChecked()),
            "model_path": self.input_model.text().strip() or "models/yolov8n-pose.pt",
            "camera_index": 0,
            "face_recognition_enabled": True,
        }
        return cfg

    # ----------------- Integration with worker -----------------
    @Slot()
    def run_detection(self):
        """Inicia ou para o worker de detecção de pose."""
        if self.worker_thread and self.worker:
            # already running -> stop
            self._stop_worker()
            self.btn_start.setText("Iniciar Detecção")
            self.status_small.setText("Parado")
            return

        # start the worker thread with configuration
        cfg = self._gather_settings()
        self.worker = DetectionWorker(config=cfg)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # connect signals
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.messages_ready.connect(self._on_messages)
        self.worker.keypoints_ready.connect(self._on_keypoints)
        self.worker.issue_ready.connect(self._on_issue)
        self.worker.fps_signal.connect(self._on_fps)
        self.worker.res_signal.connect(self._on_res)
        self.worker.finished.connect(self._on_worker_finished)

        self.worker.registration_finished.connect(self._show_registration_popup)

        self.worker_thread.started.connect(self.worker.start_worker)
        self.worker_thread.start()

        self.btn_start.setText("Parar Detecção")
        self.status_small.setText("Inicializando...")
        self.status_small.setStyleSheet("color: #f4b400;")

    def _stop_worker(self):
        """Para a thread do worker de forma segura."""
        try:
            if self.worker:
                self.worker.stop()
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait(2000)
        except Exception:
            pass
        self.worker = None
        self.worker_thread = None

    @Slot(object)
    def _on_frame_ready(self, frame_rgb):
        """
        Recebe um novo frame da câmera e o exibe na tela.

        Args:
            frame_rgb (np.array): O frame da câmera no formato RGB.
        """
        try:
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.camera_label.setPixmap(pix)
            self.lbl_res.setText(f"Res: {w}x{h}")
        except Exception as e:
            print("Erro ao desenhar frame:", e)

    @Slot(list)
    def _on_messages(self, mensagens):
        if mensagens:
            self.info_text.setPlainText("\n".join(mensagens))
        else:
            self.info_text.setPlainText("Nenhuma ação detectada.")

    @Slot(object)
    def _on_keypoints(self, keypoints):
        """
        Recebe os keypoints detectados e os envia para o canvas de análise.

        Args:
            keypoints (list): Lista de coordenadas (x, y, conf) da pose.
        """
        self._last_keypoints = keypoints
        try:
            self.coach_canvas.update_pose(keypoints)
        except Exception:
            pass

        if self.coach_canvas.target_angles:
            curr = _compute_angles(keypoints)
            ok = 0; tot = 0
            for j, tgt in self.coach_canvas.target_angles.items():
                a = curr.get(j)
                if tgt is None or a is None:
                    continue
                tot += 1
                if abs(a - tgt) <= self.coach_canvas.tolerance_ok:
                    ok += 1
            if tot > 0:
                self.metric_conform.setText(f"Conformidade de postura: {int(100*ok/tot)}%")

    @Slot(object)
    def _on_issue(self, code):
        try:
            self.coach_canvas.set_external_issue(code)
        except Exception:
            pass

    @Slot(float)
    def _on_fps(self, f):
        self.lbl_fps.setText(f"FPS: {f:.1f}")

    @Slot(str)
    def _on_res(self, s):
        self.lbl_res.setText(f"Res: {s}")

    @Slot()
    def _on_worker_finished(self):
        self.btn_start.setText("Iniciar Detecção")
        self.status_small.setText("Parado")
        self._stop_worker()

    # ----------------- original logic methods -----------------
    def _set_target_now(self):
        kp = self._get_keypoints_safe()
        if kp:
            self.coach_canvas.set_target_from_current(kp)
            self.info_text.setPlainText("Pose-alvo definida! Faça a tarefa e observe o comparativo.")
        else:
            self.info_text.setPlainText("Não foi possível definir a pose-alvo (sem keypoints no momento).")

    def _clear_target(self):
        self.coach_canvas.clear_target()
        self.info_text.setPlainText("Pose-alvo limpa.")

    def _get_keypoints_safe(self):
        try:
            if self.worker and self.worker.pose_detector and hasattr(self.worker.pose_detector, "get_keypoints"):
                return self.worker.pose_detector.get_keypoints()
        except Exception:
            pass
        return getattr(self, "_last_keypoints", None)

    @Slot(bool, str)
    def _show_registration_popup(self, success, message):
        """Exibe um pop-up com o resultado do cadastro."""
        if success:
            QMessageBox.information(self, "Cadastro de Astronauta", message)
        else:
            QMessageBox.warning(self, "Cadastro de Astronauta", message)

    def _iniciar_cadastro(self):
        nome = self.input_name.text().strip()
        if not nome:
            self.info_text.append("Digite um nome para o astronauta primeiro!")
            return

        if self.worker and self.worker.pose_detector:
            try:
                self.worker.pose_detector.iniciar_cadastro_astronauta(nome)
                self.cadastro_ativo = True
                self.info_text.append(f"Modo cadastro ativo para {nome}. Posicione o rosto e clique em Capturar.")
                return
            except Exception:
                pass

        self.info_text.append("Detector não disponível para cadastro (ainda).")

    def _capturar_cadastro(self):
        if self.cadastro_ativo:
            if self.worker and self.worker.pose_detector:
                try:
                    self.worker.process_registration_capture()

                    self.cadastro_ativo = False
                    self.input_name.clear()
                except Exception as e:
                    self.info_text.append(f"Erro inesperado durante a captura: {e}")
            else:
                self.info_text.append("Não foi possível acessar a câmera para captura.")

    # ----------------- styles -----------------
    def _card_style(self):
        return """
        QGroupBox {
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            margin-top: 6px;
            padding-top: 10px;
        }
        QGroupBox:title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 3px 0 3px;
            color: #dbe7ff;
            font-weight: bold;
        }
        """

    def _segment_button_style(self):
        return """
        QPushButton {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 6px 12px;
            color: #cfe6ff;
            font-weight: 600;
        }
        QPushButton:checked {
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #1e88e5, stop:1 #0ea5e9);
            color: white;
            border: none;
        }
        QPushButton:hover {
            background: rgba(255,255,255,0.02);
        }
        """

    def closeEvent(self, ev):
        self._stop_worker()
        super().closeEvent(ev)


