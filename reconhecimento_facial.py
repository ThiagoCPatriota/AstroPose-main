import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox

class ReconhecimentoFacial:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Carrega rostos conhecidos do diretório 'astronautas/'"""
        astronaut_dir = "astronautas"
        if not os.path.exists(astronaut_dir):
            os.makedirs(astronaut_dir)
            return
            
        encodings_file = os.path.join(astronaut_dir, "encodings.pkl")
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    print(f"Carregados {len(self.known_face_names)} astronautas cadastrados")
            except:
                print("Erro ao carregar encodings, recriando...")
                self._create_encodings_from_images(astronaut_dir)
        else:
            self._create_encodings_from_images(astronaut_dir)
    
    def _create_encodings_from_images(self, astronaut_dir):
        """Cria encodings a partir das imagens no diretório"""
        for filename in os.listdir(astronaut_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')) and filename != "encodings.pkl":
                image_path = os.path.join(astronaut_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    nome = os.path.splitext(filename)[0].replace('_', ' ')
                    self.known_face_names.append(nome)
                    self.known_face_images.append(image)
                    print(f"Carregado: {nome}")
        
        self._save_encodings()
    
    def _save_encodings(self):
        """Salva os encodings em arquivo pickle"""
        astronaut_dir = "astronautas"
        encodings_file = os.path.join(astronaut_dir, "encodings.pkl")
        
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def cadastrar_astronauta(self, frame, nome=None):
        """Cadastra um novo astronauta a partir do frame atual"""
        if nome is None:
            root = tk.Tk()
            root.withdraw()
            nome = simpledialog.askstring("Cadastro", "Digite o nome do astronauta:")
            root.destroy()
            
            if not nome:
                return False
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            messagebox.showerror("Erro", "Nenhum rosto detectado na imagem!")
            return False
        
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        astronaut_dir = "astronautas"
        if not os.path.exists(astronaut_dir):
            os.makedirs(astronaut_dir)
        
        filename = f"{nome.replace(' ', '_')}.jpg"
        image_path = os.path.join(astronaut_dir, filename)
        cv2.imwrite(image_path, frame)
        
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(nome)
        self.known_face_images.append(frame.copy())
        
        self._save_encodings()
        
        messagebox.showinfo("Sucesso", f"Astronauta {nome} cadastrado com sucesso!")
        return True
    
    def reconhecer_rostos(self, frame):
        """Reconhece rostos no frame e retorna nomes e coordenadas"""
        if not self.known_face_encodings:
            return []
            
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_coords = []
        face_confidences = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconhecido"
            confidence = 0.0
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if matches[best_match_index] and confidence > 0.6:
                    name = self.known_face_names[best_match_index]
            
            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            face_names.append(name)
            face_coords.append((left, top, right, bottom))
            face_confidences.append(confidence)
            
        return list(zip(face_names, face_coords, face_confidences))
    
    def get_astronauta_info(self, nome):
        """Retorna informações sobre um astronauta específico"""
        if nome in self.known_face_names:
            index = self.known_face_names.index(nome)
            return {
                'nome': nome,
                'imagem': self.known_face_images[index] if index < len(self.known_face_images) else None
            }
        return None
    
    def listar_astronautas(self):
        """Retorna lista de todos os astronautas cadastrados"""
        return self.known_face_names.copy()