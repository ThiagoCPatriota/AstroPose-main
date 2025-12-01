"""
Módulo de Utilitários

Este ficheiro contém funções auxiliares e de cálculo genéricas utilizadas
em várias partes da aplicação AstroPose, como cálculos de geometria para
análise de pose e listagem de capacidades da câmara.
"""
import cv2
import torch
import math
import numpy as np


KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16
}
JOINTS_FOR_ANGLES = {
    "l_elbow": (KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"]),
    "r_elbow": (KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"]),
    "l_knee": (KP["l_hip"], KP["l_knee"], KP["l_ankle"]),
    "r_knee": (KP["r_hip"], KP["r_knee"], KP["r_ankle"]),
    "l_shoulder": (KP["l_elbow"], KP["l_shoulder"], KP["l_hip"]),
    "r_shoulder": (KP["r_elbow"], KP["r_shoulder"], KP["r_hip"]),
}
MIN_CONF = 0.45

def _angle(a, b, c):
    """Calcula o ângulo em graus entre três pontos."""
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
    """Calcula o ponto médio entre dois pontos."""
    return ((p[0]+q[0])/2.0, (p[1]+q[1])/2.0)

def _clean_kp(kp):
    """Remove keypoints com baixa confiança de uma lista."""
    out = []
    for p in kp or []:
        if not p or len(p) < 3 or p[2] is None or p[2] < MIN_CONF:
            out.append(None)
        else:
            out.append(p)
    return out

def _normalize_keypoints(kp, canvas_w, canvas_h, side="left"):
    """Normaliza e escala os keypoints para serem desenhados no canvas da UI."""
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
    """Calcula os ângulos de várias articulações a partir dos keypoints."""
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
    """Escolhe automaticamente o 'sprite' do astronauta com base na pose."""
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