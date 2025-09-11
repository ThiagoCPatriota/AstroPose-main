import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class ReconhecimentoFacial:
    def __init__(self, model_name="buffalo_l", det_size=(640,640)):
        self.known_face_embeddings = []
        self.known_face_names = []

        # inicializa o modelo do InsightFace
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)

        self.load_known_faces()

    def load_known_faces(self):
        astronaut_dir = "astronautas"
        enc_file = os.path.join(astronaut_dir, "encodings.pkl")
        if os.path.exists(enc_file):
            with open(enc_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_embeddings = data["embeddings"]
                self.known_face_names = data["names"]
                print(f"Carregados {len(self.known_face_names)} astronautas")
        else:
            print("Nenhum cadastro encontrado.")

    def _save_encodings(self):
        astronaut_dir = "astronautas"
        os.makedirs(astronaut_dir, exist_ok=True)
        enc_file = os.path.join(astronaut_dir, "encodings.pkl")
        with open(enc_file, "wb") as f:
            pickle.dump({
                "embeddings": self.known_face_embeddings,
                "names": self.known_face_names
            }, f)

    def cadastrar_astronauta(self, frame, nome):
        if not nome:
            return False, "Nome não fornecido."

        faces = self.app.get(frame)
        if len(faces) != 1:
            return False, "Precisa exatamente de 1 rosto visível para cadastro."

        face = faces[0]
        emb = face.embedding / np.linalg.norm(face.embedding)

        filename = f"{nome.replace(' ', '_')}.jpg"
        cv2.imwrite(os.path.join("astronautas", filename), frame)

        self.known_face_embeddings.append(emb)
        self.known_face_names.append(nome)
        self._save_encodings()

        return True, f"Astronauta {nome} cadastrado com sucesso!"

    def reconhecer_rostos(self, frame, threshold=0.4):
        faces = self.app.get(frame)
        resultados = []

        for face in faces:
            emb = face.embedding / np.linalg.norm(face.embedding)

            if not self.known_face_embeddings:
                # se não há cadastros ainda, mostra como desconhecido
                resultados.append(("Desconhecido", face.bbox.astype(int), 0.0))
                continue

            # calcula similaridade
            sims = np.dot(self.known_face_embeddings, emb)
            idx = int(np.argmax(sims))
            score = sims[idx]

            if score > threshold:
                nome = self.known_face_names[idx]
            else:
                nome = "Desconhecido"

            resultados.append((nome, face.bbox.astype(int), float(score)))

        return resultados

    def listar_astronautas(self):
        return self.known_face_names.copy()
