"""
Detector de Cor Veicular Local com OpenCV (Custo Zero)
Classifica a cor da lataria do veículo sem necessidade de modelos em nuvem.
"""
import cv2
import numpy as np
from typing import Tuple
from .modelos import ResultadoCor

class LocalColorDetector:
    CORES_MAP = {
        "BRANCO": ([0, 0, 180], [180, 45, 255]),
        "PRETO": ([0, 0, 0], [180, 255, 55]),
        "PRATA": ([0, 0, 90], [180, 40, 180]),
        "CINZA": ([0, 0, 60], [180, 50, 150]),
        "VERMELHO_1": ([0, 70, 50], [10, 255, 255]),
        "VERMELHO_2": ([170, 70, 50], [180, 255, 255]),
        "AZUL": ([95, 70, 50], [130, 255, 255]),
        "AMARELO": ([20, 70, 70], [35, 255, 255]),
        "VERDE": ([36, 70, 50], [85, 255, 255])
    }

    @staticmethod
    def identificar_cor_predominante(caminho_imagem: str) -> Tuple[str, float]:
        """
        Analisa a região central da foto (onde fica a lataria/capô do veículo)
        e calcula a cor predominante no espaço HSV.
        """
        img = cv2.imread(caminho_imagem)
        if img is None:
            return "DESCONHECIDA", 0.0

        # Focar no centro da imagem (evita fundo, chão de asfalto e céu)
        h, w = img.shape[:2]
        crop = img[int(h * 0.3):int(h * 0.7), int(w * 0.2):int(w * 0.8)]

        # Converte para HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = crop.shape[0] * crop.shape[1]

        maior_pontuacao = 0
        cor_vencedora = "DESCONHECIDA"

        for nome_cor, (lower, upper) in LocalColorDetector.CORES_MAP.items():
            lower_np = np.array(lower, dtype="uint8")
            upper_np = np.array(upper, dtype="uint8")

            mascara = cv2.inRange(hsv, lower_np, upper_np)
            contagem = cv2.countNonZero(mascara)

            if contagem > maior_pontuacao:
                maior_pontuacao = contagem
                cor_vencedora = "VERMELHO" if "VERMELHO" in nome_cor else nome_cor

        confianca = round(maior_pontuacao / total_pixels, 2) if total_pixels > 0 else 0.0
        return cor_vencedora, confianca

    def validar_cor(self, caminho_imagem: str, cor_oficial: str = "") -> ResultadoCor:
        cor_detectada, conf = self.identificar_cor_predominante(caminho_imagem)

        oficial_clean = cor_oficial.upper().strip()
        match = False

        # Tolerâncias de cor automotiva (ex: Prata/Cinza)
        if cor_detectada == oficial_clean:
            match = True
        elif cor_detectada in ["PRATA", "CINZA"] and oficial_clean in ["PRATA", "CINZA"]:
            match = True

        return ResultadoCor(
            cor_detectada=cor_detectada,
            confianca=conf,
            match_com_cadastro=match
        )

color_detector = LocalColorDetector()
