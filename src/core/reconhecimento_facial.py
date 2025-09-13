import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class ReconhecimentoFacial:
    """
    Gerencia o cadastro e o reconhecimento de faces usando o InsightFace.
    """
    def __init__(self, model_name="buffalo_l", det_size=(640,640)):
        """
        Inicializa o modelo de análise facial.

        Args:
            model_name (str): Nome do modelo do InsightFace a ser usado.
            det_size (tuple): Tamanho da imagem para detecção.
        """
        self.known_face_embeddings = []
        self.known_face_names = []

        # inicializa o modelo do InsightFace
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)

        self.load_known_faces()

    def load_known_faces(self):
        """Carrega os embeddings e nomes de faces conhecidas de um arquivo .pkl."""
        astronaut_dir = "../../astronautas"
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
        """Salva os embeddings e nomes de faces conhecidas em um arquivo .pkl."""
        astronaut_dir = "../../astronautas"
        os.makedirs(astronaut_dir, exist_ok=True)
        enc_file = os.path.join(astronaut_dir, "encodings.pkl")
        with open(enc_file, "wb") as f:
            pickle.dump({
                "embeddings": self.known_face_embeddings,
                "names": self.known_face_names
            }, f)

    def cadastrar_astronauta(self, frame, nome):
        """
        Cadastra um novo astronauta a partir de um frame.

        Args:
            frame (np.array): O frame da câmera contendo o rosto a ser cadastrado.
            nome (str): O nome do astronauta.

        Returns:
            tuple: (success, message) indicando o resultado do cadastro.
        """
        if not nome:
            return False, "Nome não fornecido."

        faces = self.app.get(frame)
        if len(faces) != 1:
            return False, "Precisa exatamente de 1 rosto visível para cadastro."

        face = faces[0]
        emb = face.embedding / np.linalg.norm(face.embedding)

        filename = f"{nome.replace(' ', '_')}.jpg"
        cv2.imwrite(os.path.join("../../astronautas", filename), frame)

        self.known_face_embeddings.append(emb)
        self.known_face_names.append(nome)
        self._save_encodings()

        return True, f"Astronauta {nome} cadastrado com sucesso!"

    def reconhecer_rostos(self, frame, threshold=0.4):
        """
        Reconhece rostos conhecidos em um frame.

        Args:
            frame (np.array): O frame da câmera para análise.
            threshold (float): Limiar de confiança para considerar um rosto como conhecido.

        Returns:
            list: Uma lista de tuplas, cada uma contendo (nome, bbox, score).
        """
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
        """Retorna uma lista com os nomes de todos os astronautas cadastrados."""
        return self.known_face_names.copy()
