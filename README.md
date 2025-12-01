<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PySide6-informational?logo=qt" alt="PySide6">
  <img src="https://img.shields.io/badge/IA-YOLOv8%20%26%20InsightFace-orange?logo=OpenAI" alt="Machine Learning">
</p>
<h1 align="center">
  🚀 AstroPose - Coach de Postura com IA
</h1>

AstroPose é um projeto de Inteligência Artificial desenvolvido em Python que utiliza visão computacional para analisar a postura corporal em tempo real. A aplicação deteta keypoints do corpo humano para avaliar e fornecer feedback sobre diversos exercícios e posições, como agachamentos, alinhamento dos ombros e inclinação do tronco.

O projeto inclui ainda um sistema de reconhecimento facial para identificar "astronautas" (utilizadores) cadastrados e uma interface gráfica moderna construída com PySide6.

---

## ⚙️ Pré-requisitos

Antes de executar o projeto, certifique-se de ter os seguintes pré-requisitos instalados no seu sistema.

### 1. Python
- **Python 3.8 ou superior**. Pode verificar a sua versão com o comando:
  ```bash
  python --version
  ```

### 2. Ferramentas de Compilação (Obrigatório)
Algumas bibliotecas de visão computacional, como `insightface`, precisam de compilar código C++.
- **Visual Studio (com C++ Build Tools):** Instale o [Visual Studio](https://visualstudio.microsoft.com/) e, durante a instalação, marque a opção **"Desenvolvimento para desktop com C++"**.
- **CMake:** Faça o download e instale o [CMake](https://cmake.org/download/). Adicione o caminho do CMake à sua variável de ambiente `PATH` se o instalador não o fizer automaticamente.

---

## 🚀 Instalação e Execução

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/AstroPose-main.git](https://github.com/seu-usuario/AstroPose-main.git)
    cd AstroPose-main
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Instale o `onnxruntime` de acordo com o seu hardware:**
    * **Para placas de vídeo NVIDIA (recomendado para melhor desempenho):**
        ```bash
        pip install onnxruntime-gpu
        ```
    * **Para outras placas (AMD, Intel) ou para usar a CPU:**
        ```bash
        pip install onnxruntime
        ```

5.  **Execute a aplicação principal:**
    ```bash
    python main.py
    ```

---

## 📂 Estrutura do Projeto

O projeto está organizado da seguinte forma para garantir modularidade e clareza:

```
astropose/
│
├── assets/             # Ficheiros de recursos, como imagens da UI.
├── models/             # Modelos de Machine Learning (ex: yolov8n-pose.pt).
├── scripts/            # Scripts auxiliares, como a versão 'sem_placa'.
├── src/                # Diretório principal do código-fonte.
│   ├── analysis/       # Classes para análises de postura específicas.
│   ├── core/           # Núcleo da aplicação (detector, reconhecimento facial).
│   ├── ui/             # Lógica da interface gráfica (PySide6).
│   └── utils/          # Funções de utilidade e cálculos.
│
├── main.py             # Ponto de entrada para executar a aplicação.
├── requirements.txt    # Lista de dependências Python.
└── README.md           # Este ficheiro.
```