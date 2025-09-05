# reconhecimento_facial.py
import cv2
import face_recognition
import numpy as np
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
import re

ID_PREFIX = "AST"

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _safe_name(name: str) -> str:
    return "_".join(name.strip().split())

def _parse_dir_name(dirname: str):
    """
    Aceita diretórios no formato:
      - "<safe_name>"
      - "<safe_name>__AST-0007"
    Retorna (name, person_id ou None)
    """
    if "__" in dirname:
        left, right = dirname.split("__", 1)
        if re.fullmatch(rf"{ID_PREFIX}-\d+", right):
            return left.replace("_", " "), right
    return dirname.replace("_", " "), None

class ReconhecimentoFacial:
    def __init__(self):
        # paralelos e indexados 1:1
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        # contador para novo ID
        self.last_id_num = 0
        self.known_face_images = []  # opcional (amostra visual)

        self.load_known_faces()

    # ---------------- Persistência ----------------
    def load_known_faces(self):
        """Carrega encodings e metadados de 'astronautas/encodings.pkl'."""
        astronaut_dir = _ensure_dir("astronautas")
        encodings_file = os.path.join(astronaut_dir, "encodings.pkl")

        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)

                self.known_face_encodings = data.get('encodings', [])
                self.known_face_names = data.get('names', [])
                self.known_face_ids = data.get('ids', [])
                self.last_id_num = int(data.get('last_id_num', 0))

                # retrocompat: se não tiver ids, criar temporários
                if not self.known_face_ids or len(self.known_face_ids) != len(self.known_face_names):
                    print("[AVISO] Encodings sem IDs. Recriando a partir de imagens...")
                    self._create_encodings_from_images(astronaut_dir)
                else:
                    # ajustar last_id_num pelo máximo observado
                    for pid in self.known_face_ids:
                        m = re.search(rf"{ID_PREFIX}-(\d+)", str(pid))
                        if m:
                            self.last_id_num = max(self.last_id_num, int(m.group(1)))
                    print(f"Carregados {len(self.known_face_names)} amostras de rostos. last_id={self.last_id_num}")
                return
            except Exception as e:
                print(f"[AVISO] Falha ao carregar encodings: {e}. Recriando a partir das imagens...")

        self._create_encodings_from_images(astronaut_dir)

    def _create_encodings_from_images(self, astronaut_dir):
        """Regenera encodings varrendo subpastas por pessoa; tenta preservar IDs a partir do nome da pasta."""
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_ids.clear()
        self.known_face_images.clear()

        # varrer pastas
        for item in os.listdir(astronaut_dir):
            path = os.path.join(astronaut_dir, item)
            if not os.path.isdir(path):
                continue

            person_name, pid = _parse_dir_name(item)
            if not pid:
                pid = self._next_id()  # gera um novo ID se não existir

            for filename in os.listdir(path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and "contact_sheet" not in filename.lower():
                    image_path = os.path.join(path, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encs = face_recognition.face_encodings(image)
                        if encs:
                            self.known_face_encodings.append(encs[0])
                            self.known_face_names.append(person_name)
                            self.known_face_ids.append(pid)
                            self.known_face_images.append(image)
                            # print(f"Carregado: {person_name} [{pid}] ({filename})")
                    except Exception as e:
                        print(f"[AVISO] Erro ao processar {filename}: {e}")

        self._save_encodings()

    def _save_encodings(self):
        """Escrita atômica do pickle (encodings + names + ids + last_id_num)."""
        astronaut_dir = _ensure_dir("astronautas")
        encodings_file = os.path.join(astronaut_dir, "encodings.pkl")
        tmp_file = encodings_file + ".tmp"
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'ids': self.known_face_ids,
            'last_id_num': self.last_id_num,
        }
        with open(tmp_file, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_file, encodings_file)

    # ---------------- IDs ----------------
    def _next_id(self) -> str:
        self.last_id_num += 1
        return f"{ID_PREFIX}-{self.last_id_num:04d}"

    def new_person_id(self) -> str:
        """Gera e reserva um novo ID (não usa nenhum estado além do contador)."""
        return self._next_id()

    def ids_by_name(self, name: str):
        """Retorna conjunto de IDs já vistos para este nome (pode haver mais de um)."""
        name = name.strip()
        ids = set()
        for n, pid in zip(self.known_face_names, self.known_face_ids):
            if n == name:
                ids.add(pid)
        return sorted(list(ids))

    # ---------------- Cadastro (multi-foto) ----------------
    def finalize_person(self, nome: str, person_id: str, encodings, saved_paths, overwrite=False):
        """
        Conclui cadastro: adiciona encodings para (nome, person_id) e salva.
        Se overwrite=True, remove encodings anteriores desse mesmo ID.
        """
        if overwrite:
            # remove entradas do mesmo ID
            keep_enc = []
            keep_nam = []
            keep_ids = []
            for e, n, pid in zip(self.known_face_encodings, self.known_face_names, self.known_face_ids):
                if pid != person_id:
                    keep_enc.append(e); keep_nam.append(n); keep_ids.append(pid)
            self.known_face_encodings = keep_enc
            self.known_face_names = keep_nam
            self.known_face_ids = keep_ids

        count_added = 0
        for enc in encodings:
            self.known_face_encodings.append(enc)
            self.known_face_names.append(nome)
            self.known_face_ids.append(person_id)
            count_added += 1

        # amostra visual (opcional)
        try:
            if saved_paths:
                img0 = face_recognition.load_image_file(saved_paths[-1])
                self.known_face_images.append(img0)
        except Exception:
            pass

        self._save_encodings()
        return {"ok": True, "msg": f"Astronauta '{nome}' [{person_id}] cadastrado com {count_added} fotos.", "id": person_id}

    # ---------------- Reconhecimento ----------------
    def reconhecer_rostos(self, frame, tolerance=0.6):
        """
        Retorna lista de tuplas:
            (name, person_id, (left, top, right, bottom), score[0-1])
        """
        if not self.known_face_encodings:
            return []

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        results = []
        for enc, loc in zip(face_encodings, face_locations):
            if len(self.known_face_encodings) == 0:
                continue

            dists = face_recognition.face_distance(self.known_face_encodings, enc)
            idx = int(np.argmin(dists))
            best_dist = float(dists[idx])
            score = max(0.0, (tolerance - best_dist) / tolerance)

            name = "Desconhecido"
            pid = "—"
            if best_dist <= tolerance:
                name = self.known_face_names[idx]
                pid = self.known_face_ids[idx]

            top, right, bottom, left = loc
            top *= 4; right *= 4; bottom *= 4; left *= 4
            results.append((name, pid, (left, top, right, bottom), score))

        return results

    # ---------------- Contact sheet ----------------
    def make_contact_sheet(self, paths, out_dir, safe_name, cols=4, thumb=360, title=None):
        """Gera uma página com miniaturas (JPG e PDF)."""
        imgs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(img)
            except Exception:
                pass
        if not imgs:
            return None, None

        import math as _math
        rows = int(_math.ceil(len(imgs) / float(cols)))
        w = cols * thumb
        h = rows * thumb + 80
        sheet = Image.new("RGB", (w, h), (245, 245, 245))

        draw = ImageDraw.Draw(sheet)
        title = title or f"Astronauta: {safe_name.replace('_', ' ')}"
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        draw.text((16, 16), title, fill=(20, 20, 20), font=font)

        y0 = 64
        for i, img in enumerate(imgs):
            t = img.copy()
            t.thumbnail((thumb, thumb))
            r = i // cols
            c = i % cols
            x = c * thumb
            y = y0 + r * thumb
            sheet.paste(t, (x, y))

        page_jpg = os.path.join(out_dir, f"{safe_name}_contact_sheet.jpg")
        sheet.save(page_jpg, "JPEG", quality=90)

        page_pdf = os.path.join(out_dir, f"{safe_name}_contact_sheet.pdf")
        sheet.convert("RGB").save(page_pdf, "PDF", resolution=200.0)

        return page_jpg, page_pdf

    # ---------------- Utilidades ----------------
    def get_astronauta_info(self, nome):
        """Retorna info básica (não indexado por ID)."""
        for n, img in zip(self.known_face_names, self.known_face_images):
            if n == nome:
                return {'nome': nome, 'imagem': img}
        return None

    def listar_astronautas(self):
        """Lista nomes (pode haver repetidos)."""
        return self.known_face_names.copy()
