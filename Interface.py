import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import cv2
from detector import PoseDetector
import time
import math

# ==========================
# AstroPose - UI com "Bonequinho Coach"
# ==========================

# ----- Sprites -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_DIR = os.path.join(BASE_DIR, "img")
SPRITE_MAP = {
    "Neutro":     os.path.join(SPRITE_DIR, "neutral.png"),
    "Joinha":     os.path.join(SPRITE_DIR, "thumbsup.png"),
    "Apontando":  os.path.join(SPRITE_DIR, "point.png"),
    "T-pose":     os.path.join(SPRITE_DIR, "tpose.png"),
}
SPRITE_CHOICES = ["Auto", "Neutro", "Joinha", "Apontando", "T-pose"]
MIN_CONF = 0.45

# ----- Keypoints / Esqueleto -----
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16
}
BONES = [
    (KP["l_shoulder"], KP["r_shoulder"]),
    (KP["l_shoulder"], KP["l_elbow"]), (KP["l_elbow"], KP["l_wrist"]),
    (KP["r_shoulder"], KP["r_elbow"]), (KP["r_elbow"], KP["r_wrist"]),
    (KP["l_hip"], KP["r_hip"]),
    (KP["l_shoulder"], KP["l_hip"]), (KP["r_shoulder"], KP["r_hip"]),
    (KP["l_hip"], KP["l_knee"]), (KP["l_knee"], KP["l_ankle"]),
    (KP["r_hip"], KP["r_knee"]), (KP["r_knee"], KP["r_ankle"]),
]
JOINTS_FOR_ANGLES = {
    "l_elbow": (KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"]),
    "r_elbow": (KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"]),
    "l_knee": (KP["l_hip"], KP["l_knee"], KP["l_ankle"]),
    "r_knee": (KP["r_hip"], KP["r_knee"], KP["r_ankle"]),
    "l_shoulder": (KP["l_elbow"], KP["l_shoulder"], KP["l_hip"]),
    "r_shoulder": (KP["r_elbow"], KP["r_shoulder"], KP["r_hip"]),
}

# ----- Helpers -----
def _angle(a, b, c):
    try:
        ax, ay = a[:2]; bx, by = b[:2]; cx, cy = c[:2]
        v1 = (ax - bx, ay - by); v2 = (cx - bx, cy - by)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0: return None
        cosang = max(-1.0, min(1.0, dot/(n1*n2)))
        return math.degrees(math.acos(cosang))
    except Exception:
        return None

def _midpoint(p, q):
    return ((p[0]+q[0])/2.0, (p[1]+q[1])/2.0)

def _clean_kp(kp):
    out = []
    for p in kp or []:
        if not p or len(p) < 3 or p[2] is None or p[2] < MIN_CONF:
            out.append(None)
        else:
            out.append(p)
    return out

def _normalize_keypoints(kp, canvas_w, canvas_h, side="left"):
    try:
        kp = _clean_kp(kp)
        if not kp or all(p is None for p in kp):
            return None
        ls = kp[KP["l_shoulder"]] if KP["l_shoulder"] < len(kp) else None
        rs = kp[KP["r_shoulder"]] if KP["r_shoulder"] < len(kp) else None
        lh = kp[KP["l_hip"]] if KP["l_hip"] < len(kp) else None
        rh = kp[KP["r_hip"]] if KP["r_hip"] < len(kp) else None

        if ls and rs:
            cx, cy = _midpoint(ls, rs)
            shoulder_dist = max(1.0, math.hypot(rs[0]-ls[0], rs[1]-ls[1]))
        else:
            vs = [(p[0], p[1]) for p in kp if p]
            cx = sum(x for x,_ in vs)/len(vs)
            cy = sum(y for _,y in vs)/len(vs)
            shoulder_dist = 100.0

        scale_ref = shoulder_dist
        if lh and rh and ls and rs:
            shoulder_mid = _midpoint(ls, rs)
            hip_mid = _midpoint(lh, rh)
            torso_len = math.hypot(shoulder_mid[0]-hip_mid[0], shoulder_mid[1]-hip_mid[1])
            if torso_len > 1e-3:
                scale_ref = max(scale_ref, torso_len*1.2)

        box = 300
        scale = (box/2.4) / scale_ref
        origin_x = canvas_w*0.25 if side == "left" else canvas_w*0.75
        origin_y = canvas_h*0.58

        out = []
        for p in kp:
            if not p:
                out.append(None)
            else:
                x = origin_x + (p[0]-cx)*scale
                y = origin_y + (p[1]-cy)*scale
                out.append((x, y, p[2]))
        return out
    except Exception:
        return None

def _compute_angles(kp):
    kp = _clean_kp(kp)
    out = {}
    for name, (a,b,c) in JOINTS_FOR_ANGLES.items():
        pa = kp[a] if a < len(kp) else None
        pb = kp[b] if b < len(kp) else None
        pc = kp[c] if c < len(kp) else None
        out[name] = _angle(pa, pb, pc) if (pa and pb and pc) else None
    try:
        ls = kp[KP["l_shoulder"]] if KP["l_shoulder"] < len(kp) else None
        rs = kp[KP["r_shoulder"]] if KP["r_shoulder"] < len(kp) else None
        lh = kp[KP["l_hip"]] if KP["l_hip"] < len(kp) else None
        rh = kp[KP["r_hip"]] if KP["r_hip"] < len(kp) else None
        if ls and rs and lh and rh:
            shoulder_mid = (_midpoint(ls, rs)[0], _midpoint(ls, rs)[1], 1)
            hip_mid = (_midpoint(lh, rh)[0], _midpoint(lh, rh)[1], 1)
            v = (shoulder_mid[0]-hip_mid[0], shoulder_mid[1]-hip_mid[1])
            dot = v[1]*(-1); n = math.hypot(*v)
            tilt = None if n == 0 else math.degrees(math.acos(max(-1,min(1,dot/n))))
            out["trunk_tilt"] = tilt
        elif ls and rs:
            v = (rs[0]-ls[0], rs[1]-ls[1])
            out["trunk_tilt"] = abs(math.degrees(math.atan2(v[1], v[0])))
        else:
            out["trunk_tilt"] = None
    except Exception:
        out["trunk_tilt"] = None
    return out

def _choose_sprite_auto(kp):
    try:
        kp = _clean_kp(kp)
        ls, rs = kp[KP["l_shoulder"]], kp[KP["r_shoulder"]]
        le, re = kp[KP["l_elbow"]], kp[KP["r_elbow"]]
        lw, rw = kp[KP["l_wrist"]], kp[KP["r_wrist"]]
        if all(p for p in [ls, rs, le, re, lw, rw]):
            left_horiz  = abs((lw[1]-ls[1])) < 0.15*abs(lw[0]-ls[0]) if abs(lw[0]-ls[0]) > 1 else False
            right_horiz = abs((rw[1]-rs[1])) < 0.15*abs(rw[0]-rs[0]) if abs(rw[0]-rs[0]) > 1 else False
            elbows_straight = (_angle(ls, le, lw) or 180) > 150 and (_angle(rs, re, rw) or 180) > 150
            if left_horiz and right_horiz and elbows_straight:
                return "T-pose"
        if rs and re and rw:
            if rw[1] < rs[1] and (_angle(rs, re, rw) or 180) < 110:
                return "Joinha"
        if rs and rw and (_angle(rs, re, rw) or 180) > 150:
            if abs(rw[1]-rs[1]) < 0.2*abs(rw[0]-rs[0]):
                return "Apontando"
        return "Neutro"
    except Exception:
        return "Neutro"


# ==========================
# Canvas do Coach
# ==========================
class CoachView:
    def __init__(self, parent, width=400, height=380, colors=None):
        self.canvas = tk.Canvas(parent, width=width, height=height, highlightthickness=0, bg=colors["card"])
        self.canvas.pack(fill="both", expand=True)
        self.colors = colors
        self.width = width
        self.height = height

        self.tolerance_ok = 14
        self.tolerance_warn = 28

        self.last_current_kp = None
        self.target_angles = None
        self.target_label = None
        self.target_raw_kp = None

        self.sprite_choice = "Auto"
        self.sprite_path = None
        self.sprite_img_ref = None

        # NOVO: erro dominante vindo do detector
        self.external_issue = None

    # --- API ---
    def set_sprite_choice(self, choice):
        self.sprite_choice = choice
        if choice in SPRITE_MAP and os.path.exists(SPRITE_MAP[choice]):
            self.sprite_path = SPRITE_MAP[choice]
        elif choice != "Auto":
            self.sprite_path = None
        self._redraw()

    def set_external_issue(self, issue):
        self.external_issue = issue

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

    def set_target_from_current(self, raw_kp):
        if not raw_kp:
            return
        self.target_raw_kp = raw_kp
        self.target_angles = _compute_angles(raw_kp)
        self.target_label = _choose_sprite_auto(raw_kp) or "Neutro"
        if self.sprite_choice == "Auto":
            p = SPRITE_MAP.get(self.target_label)
            self.sprite_path = p if p and os.path.exists(p) else None
        self._redraw()

    def clear_target(self):
        self.target_angles = None
        self.target_label = None
        self.target_raw_kp = None
        if self.sprite_choice == "Auto":
            self.sprite_path = None
        self._redraw()

    def update_pose(self, raw_kp):
        self.last_current_kp = raw_kp
        if self.sprite_choice == "Auto":
            self._choose_sprite_auto_corrective()
        self._redraw()

    # --- escolha do sprite ---
    def _choose_sprite_auto_corrective(self):
        # 1) prioridade: erro reportado pelo detector
        name_ext = self._issue_to_sprite()
        if name_ext and self.sprite_choice == "Auto":
            p = SPRITE_MAP.get(name_ext)
            if p and os.path.exists(p):
                self.sprite_path = p
                return

        # 2) sem erro explícito, usa alvo/heurística
        if self.target_angles is None or self.last_current_kp is None:
            name = _choose_sprite_auto(self.last_current_kp or [])
            p = SPRITE_MAP.get(name)
            self.sprite_path = p if p and os.path.exists(p) else None
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

    # --- desenho ---
    def _draw_skeleton(self, kp_norm, color="#e5e7eb", thickness=3, highlight_bones=None):
        if not kp_norm:
            return
        MAX_LEN = 260
        for (i, j) in BONES:
            pi = kp_norm[i] if i < len(kp_norm) else None
            pj = kp_norm[j] if j < len(kp_norm) else None
            if not pi or not pj:
                continue
            if math.hypot(pi[0] - pj[0], pi[1] - pj[1]) > MAX_LEN:
                continue
            col = self.colors["danger"] if (highlight_bones and ((i, j) in highlight_bones or (j, i) in highlight_bones)) else color
            self.canvas.create_line(pi[0], pi[1], pj[0], pj[1], fill=col, width=thickness, capstyle="round")
        for p in kp_norm:
            if not p:
                continue
            self.canvas.create_oval(p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4, fill=color, outline="")

    def _draw_sprite_target(self):
        if not self.sprite_path or not os.path.exists(self.sprite_path):
            return
        try:
            img = Image.open(self.sprite_path).convert("RGBA")
            max_w, max_h = 300, 340
            w, h = img.size
            scale = min(max_w / float(w), max_h / float(h))
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.sprite_img_ref = ImageTk.PhotoImage(img)
            x = self.width * 0.25
            y = self.height * 0.58
            self.canvas.create_image(x, y, image=self.sprite_img_ref)
        except Exception:
            self.sprite_img_ref = None

    def _redraw(self):
        self.canvas.delete("all")
        self.canvas.create_text(self.width * 0.25, 28, text="POSE-ALVO", fill=self.colors["muted"], font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(self.width * 0.75, 28, text="VOCÊ", fill=self.colors["muted"], font=("Segoe UI", 11, "bold"))

        self._draw_sprite_target()

        if not self.last_current_kp:
            self.canvas.create_text(self.width / 2, self.height / 2, text="Aguardando pose…", fill=self.colors["muted"], font=("Segoe UI", 12))
            return

        cur_norm = _normalize_keypoints(self.last_current_kp, self.width, self.height, side="right")

        highlight_bones = set()
        conf_ok = 0; total = 0
        if self.target_angles:
            curr = _compute_angles(self.last_current_kp)
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
            if total > 0:
                conf = int(100 * (conf_ok / max(1, total)))
                txt = [f"Conformidade: {conf}%"]
                for j, d in sorted(diffs.items(), key=lambda kv: -kv[1])[:3]:
                    txt.append(f"{j.replace('_',' ').title()}: {d:.0f}°")
                self.canvas.create_text(self.width / 2, self.height - 16, text="  •  ".join(txt), fill=self.colors["muted"], font=("Segoe UI", 10))

        self._draw_skeleton(cur_norm, color="#e5e7eb", thickness=3, highlight_bones=highlight_bones)


# ==========================
# UI Principal
# ==========================
class PoseDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AstroPose — Coach de Postura em Microgravidade")
        self.root.geometry("1180x760")
        self.root.minsize(1024, 680)

        self.COLORS = {
            "bg": "#0f172a", "panel": "#111827", "card": "#0b1220",
            "muted": "#94a3b8", "text": "#e5e7eb", "accent": "#60a5fa",
            "accent2": "#22d3ee", "danger": "#f87171", "success": "#34d399",
            "warning": "#fbbf24", "border": "#1f2937",
        }
        self._configure_style()

        self.info_text = tk.StringVar(value="Informações:\nNenhuma ação detectada.")
        self.status_text = tk.StringVar(value="Aguardando início")
        self.fps_text = tk.StringVar(value="FPS: —")
        self.res_text = tk.StringVar(value="Res: —")
        self.mode_text = tk.StringVar(value="Modo: Terra↔Microg (visual)")
        self.loading_text = "AstroPose"
        self.letter_index = 0
        self._animating = False

        self.root.configure(bg=self.COLORS["bg"])
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)

        self._build_header()
        self._build_camera_panel()
        self._build_sidebar_with_coach()
        self._build_statusbar()

        self.pose_detector = None

    def _configure_style(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except: pass
        style.configure("TFrame", background=self.COLORS["bg"])
        style.configure("Card.TFrame", background=self.COLORS["card"], borderwidth=1, relief="ridge")
        style.configure("Panel.TFrame", background=self.COLORS["panel"])
        style.configure("Header.TLabel", background=self.COLORS["bg"], foreground=self.COLORS["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subheader.TLabel", background=self.COLORS["bg"], foreground=self.COLORS["muted"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.COLORS["card"], foreground=self.COLORS["text"], font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background=self.COLORS["card"], foreground=self.COLORS["text"], font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.COLORS["card"], foreground=self.COLORS["muted"], font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=self.COLORS["panel"], foreground=self.COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.map("Accent.TButton", background=[("!disabled", self.COLORS["accent"])], foreground=[("!disabled", "#0b1020")])
        style.configure("Danger.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.map("Danger.TButton", background=[("!disabled", self.COLORS["danger"])], foreground=[("!disabled", "#0b1020")])
        style.configure("TNotebook", background=self.COLORS["panel"])
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"),
                        background=self.COLORS["panel"], foreground=self.COLORS["muted"])
        style.map("TNotebook.Tab", background=[("selected", self.COLORS["card"])], foreground=[("selected", self.COLORS["text"])])

    def _build_header(self):
        header = ttk.Frame(self.root, style="TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        header.grid_columnconfigure(2, weight=0)

        self.title_label = ttk.Label(header, text="AstroPose", style="Header.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w", padx=20, pady=(14, 4))
        subtitle = ttk.Label(header, text="Coach de Postura em Microgravidade — Demo (ISS / Fatores Humanos)", style="Subheader.TLabel")
        subtitle.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 12))

        self.status_pill = tk.Label(header, textvariable=self.status_text, bg=self.COLORS["panel"], fg=self.COLORS["muted"],
                                    font=("Segoe UI", 10, "bold"), padx=12, pady=6, bd=0, relief="flat")
        self.status_pill.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 12), pady=12)

        self.start_button = ttk.Button(header, text="Iniciar Detecção", style="Accent.TButton", command=self.run_detection)
        self.start_button.grid(row=0, column=2, rowspan=2, sticky="e", padx=20, pady=12)

    def _build_camera_panel(self):
        camera_wrap = ttk.Frame(self.root, style="Panel.TFrame")
        camera_wrap.grid(row=1, column=0, padx=(16, 8), pady=(8, 8), sticky="nsew")
        camera_wrap.grid_rowconfigure(1, weight=1)
        camera_wrap.grid_columnconfigure(0, weight=1)

        cam_title = ttk.Label(camera_wrap, text="Visualização da Câmera", style="Title.TLabel")
        cam_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        cam_card = ttk.Frame(camera_wrap, style="Card.TFrame")
        cam_card.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        cam_card.grid_rowconfigure(0, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)

        self.label = tk.Label(cam_card, bg="black", bd=0)
        self.label.grid(row=0, column=0, sticky="nsew")

        self.loading_label = tk.Label(cam_card, text="", font=("Segoe UI", 36, "bold"),
                                      fg=self.COLORS["accent2"], bg=self.COLORS["card"])

    def _build_sidebar_with_coach(self):
        sidebar = ttk.Frame(self.root, style="Panel.TFrame")
        sidebar.grid(row=1, column=1, padx=(8, 16), pady=(8, 8), sticky="nsew")
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        side_title = ttk.Label(sidebar, text="Painel de Informações", style="Title.TLabel")
        side_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        notebook = ttk.Notebook(sidebar)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        # --- Resumo ---
        tab_resumo = ttk.Frame(notebook, style="TFrame")
        notebook.add(tab_resumo, text="Resumo")
        resumo_card = ttk.Frame(tab_resumo, style="Card.TFrame")
        resumo_card.pack(fill="both", expand=True)
        lbl_head = ttk.Label(resumo_card, text="Mensagens / Ações Detectadas", style="Title.TLabel")
        lbl_head.pack(anchor="w", padx=12, pady=(12, 8))
        self.info_label = ttk.Label(resumo_card, textvariable=self.info_text, style="Body.TLabel", justify="left", anchor="nw", wraplength=380)
        self.info_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # --- Coach ---
        tab_coach = ttk.Frame(notebook, style="TFrame")
        notebook.add(tab_coach, text="Coach (Boneco)")
        coach_card = ttk.Frame(tab_coach, style="Card.TFrame")
        coach_card.pack(fill="both", expand=True)
        coach_title = ttk.Label(coach_card, text="Pose-alvo vs Você", style="Title.TLabel")
        coach_title.pack(anchor="w", padx=12, pady=(12, 4))

        btns = ttk.Frame(coach_card, style="TFrame")
        btns.pack(fill="x", padx=12, pady=(0,8))
        self.btn_set_target = ttk.Button(btns, text="Definir pose-alvo (agora)", style="Accent.TButton", command=self._set_target_now)
        self.btn_clear_target = ttk.Button(btns, text="Limpar pose-alvo", style="Danger.TButton", command=self._clear_target)
        self.btn_set_target.pack(side="left", padx=(0,8))
        self.btn_clear_target.pack(side="left")

        self.coach_view = CoachView(coach_card, width=380, height=380, colors=self.COLORS)

        # --- Métricas ---
        tab_metrics = ttk.Frame(notebook, style="TFrame")
        notebook.add(tab_metrics, text="Métricas")
        metrics_card = ttk.Frame(tab_metrics, style="Card.TFrame")
        metrics_card.pack(fill="both", expand=True)
        m_title = ttk.Label(metrics_card, text="Indicadores (MVP)", style="Title.TLabel")
        m_title.pack(anchor="w", padx=12, pady=(12, 8))
        self.metric_conform = ttk.Label(metrics_card, text="Conformidade de postura: —", style="Body.TLabel")
        self.metric_latency = ttk.Label(metrics_card, text="Latência média (estimada): —", style="Body.TLabel")
        self.metric_epi = ttk.Label(metrics_card, text="Alertas EPI: —", style="Body.TLabel")
        self.metric_conform.pack(anchor="w", padx=12, pady=2)
        self.metric_latency.pack(anchor="w", padx=12, pady=2)
        self.metric_epi.pack(anchor="w", padx=12, pady=(2, 12))

        note = ttk.Label(metrics_card, text="*Defina uma pose-alvo para ativar a comparação articulada.\n*Ângulos: ombros, cotovelos, joelhos e inclinação do tronco.",
                         style="Muted.TLabel", wraplength=380, justify="left")
        note.pack(anchor="w", padx=12, pady=(0, 12))

        self.close_button = ttk.Button(sidebar, text="Fechar", style="Danger.TButton", command=self.close_window)
        self.close_button.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    def _build_statusbar(self):
        status = ttk.Frame(self.root, style="Panel.TFrame")
        status.grid(row=2, column=0, columnspan=2, sticky="ew")
        for i in range(4): status.grid_columnconfigure(i, weight=1)
        ttk.Label(status, textvariable=self.fps_text, style="Status.TLabel").grid(row=0, column=0, sticky="w", padx=16, pady=6)
        ttk.Label(status, textvariable=self.res_text, style="Status.TLabel").grid(row=0, column=1, sticky="w", padx=16, pady=6)
        ttk.Label(status, textvariable=self.mode_text, style="Status.TLabel").grid(row=0, column=2, sticky="w", padx=16, pady=6)
        ttk.Label(status, text="© 2025 AstroPose — Demo", style="Status.TLabel").grid(row=0, column=3, sticky="e", padx=16, pady=6)

    # ---------- ações ----------
    def run_detection(self):
        self._start_title_animation()
        self.status_text.set("Inicializando...")
        self.status_pill.configure(bg=self.COLORS["panel"], fg=self.COLORS["warning"])
        self.loading_label.config(text="")
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self.pose_detector = PoseDetector()
        threading.Thread(target=self.start_pose_detection, daemon=True).start()

    def _start_title_animation(self):
        self.letter_index = 0; self._animating = True; self._animate_title_step()
    def _animate_title_step(self):
        if not self._animating: return
        text = self.loading_text[: self.letter_index + 1]
        self.title_label.config(text=text, foreground=self._next_accent_color())
        self.letter_index += 1
        if self.letter_index < len(self.loading_text):
            self.root.after(120, self._animate_title_step)
        else:
            self.root.after(350, lambda: self.title_label.config(foreground=self.COLORS["text"]))
            self._animating = False
    def _next_accent_color(self):
        return [self.COLORS["accent"], self.COLORS["accent2"], self.COLORS["success"], self.COLORS["warning"]][self.letter_index % 4]

    def _set_target_now(self):
        kp = self._get_keypoints_safe()
        if kp:
            self.coach_view.set_target_from_current(kp)
            self.info_text.set("Pose-alvo definida! Faça a tarefa e observe o comparativo.")
        else:
            self.info_text.set("Não foi possível definir a pose-alvo (sem keypoints no momento).")

    def _clear_target(self):
        self.coach_view.clear_target()
        self.info_text.set("Pose-alvo limpa.")

    def _get_keypoints_safe(self):
        try:
            if hasattr(self.pose_detector, "get_keypoints"):
                return self.pose_detector.get_keypoints()
        except Exception:
            pass
        return getattr(self, "_last_keypoints", None)

    def start_pose_detection(self):
        try:
            prev_time = time.time()
            while True:
                got_kp = False
                try:
                    result = self.pose_detector.detectar_pose()
                    if isinstance(result, tuple) and len(result) == 3:
                        mensagens, annotated_frame, keypoints = result
                        self._last_keypoints = keypoints
                        got_kp = True
                    else:
                        mensagens, annotated_frame = result
                        keypoints = None
                except ValueError:
                    mensagens, annotated_frame = self.pose_detector.detectar_pose()
                    keypoints = None

                if mensagens:
                    self.info_text.set("\n".join(mensagens))
                else:
                    self.info_text.set("Nenhuma ação detectada.")

                kp_use = keypoints if got_kp else self._get_keypoints_safe()
                if kp_use is not None:
                    self.coach_view.update_pose(kp_use)
                    # NOVO: envia o erro dominante para o boneco
                    if hasattr(self.pose_detector, "get_issue_code"):
                        self.coach_view.set_external_issue(self.pose_detector.get_issue_code())

                    if self.coach_view.target_angles:
                        curr = _compute_angles(kp_use)
                        ok = 0; tot = 0
                        for j, tgt in self.coach_view.target_angles.items():
                            a = curr.get(j)
                            if tgt is None or a is None:
                                continue
                            tot += 1
                            if abs(a - tgt) <= self.coach_view.tolerance_ok:
                                ok += 1
                        if tot > 0:
                            self.metric_conform.config(text=f"Conformidade de postura: {int(100*ok/tot)}%")

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, _ = annotated_frame_rgb.shape
                self.res_text.set(f"Res: {w}x{h}")
                img = Image.fromarray(annotated_frame_rgb)
                target_w = max(640, self.label.winfo_width() or 640)
                target_h = max(360, self.label.winfo_height() or 360)
                img = img.resize((target_w, target_h))
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

                now = time.time(); dt = now - prev_time; prev_time = now
                if dt > 0: self.fps_text.set(f"FPS: {1.0/dt:.1f}")

                self.status_text.set("Detectando...")
                self.status_pill.configure(bg=self.COLORS["panel"], fg=self.COLORS["success"])
                self.loading_label.place_forget()

                time.sleep(0.02)
                self.root.update()

        except Exception as e:
            try:
                if self.pose_detector: self.pose_detector.liberarRecursos()
            except Exception:
                pass
            self.loading_label.place_forget()
            self.status_text.set("Erro na detecção")
            self.status_pill.configure(bg=self.COLORS["panel"], fg=self.COLORS["danger"])
            self.info_text.set("Erro na detecção: " + str(e))

    def close_window(self):
        try:
            if self.pose_detector: self.pose_detector.liberarRecursos()
        except Exception:
            pass
        self.root.quit(); self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------- Entry point ----------
def main():
    gui = PoseDetectionGUI()
    gui.run()

if __name__ == "__main__":
    main()
