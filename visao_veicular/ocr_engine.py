"""
Motor de OCR Local com OpenCV e Pré-Processamento de Vidros Automotivos
Desenvolvido especificamente para tratar marcações pontilhadas (RAM, Fiat, Jeep) e reflexos.

REGRA DE SEGURANÇA: se o OCR não estiver disponível ou não ler nada, o resultado é
SEMPRE "não lido / conferir manualmente". O sistema nunca supõe que o chassi confere.
"""
import re
import cv2
import numpy as np
from typing import Tuple, List, Optional
from .modelos import ResultadoOCR


class LocalOCREngine:
    def __init__(self):
        self._paddle_ocr = None
        self._api = None  # "v3" (PaddleOCR 3.x, .predict) ou "v2" (PaddleOCR 2.x, .ocr)
        self._tentou_carregar_paddle = False
        self.erro_carregamento: Optional[str] = None

    def _obter_ocr(self):
        if self._tentou_carregar_paddle:
            return self._paddle_ocr
        self._tentou_carregar_paddle = True
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            self.erro_carregamento = f"PaddleOCR não instalado: {e}"
            print(f"[Aviso OCR] {self.erro_carregamento}. Chassi ficará para conferência manual.")
            return None

        # PaddleOCR 3.x (parâmetros novos)
        try:
            self._paddle_ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
            self._api = "v3"
            return self._paddle_ocr
        except (TypeError, ValueError):
            pass  # provavelmente PaddleOCR 2.x
        except Exception as e:
            self.erro_carregamento = f"Falha ao iniciar PaddleOCR 3.x: {e}"

        # PaddleOCR 2.x (parâmetros antigos)
        try:
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._api = "v2"
            self.erro_carregamento = None
        except Exception as e:
            self.erro_carregamento = self.erro_carregamento or f"Falha ao iniciar PaddleOCR: {e}"
            self._paddle_ocr = None

        if self._paddle_ocr is None:
            print(f"[Aviso OCR] {self.erro_carregamento}. Chassi ficará para conferência manual.")
        return self._paddle_ocr

    def _executar_ocr(self, ocr, img: np.ndarray) -> Tuple[List[str], List[float]]:
        """Executa o OCR e normaliza a saída das versões 2.x e 3.x para (textos, confiancas)."""
        # PaddleOCR espera 3 canais; a imagem pré-processada é em tons de cinza
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        textos: List[str] = []
        confs: List[float] = []
        if self._api == "v3":
            for res in ocr.predict(img) or []:
                try:
                    textos.extend(res["rec_texts"])
                    confs.extend(float(s) for s in res["rec_scores"])
                except (KeyError, TypeError):
                    continue
        else:
            resultados = ocr.ocr(img, cls=True)
            if resultados and resultados[0]:
                for linha in resultados[0]:
                    textos.append(linha[1][0])
                    confs.append(float(linha[1][1]))
        return textos, confs

    @staticmethod
    def preprocessar_vidro_pontilhado(img_bgr: np.ndarray) -> np.ndarray:
        """
        Aplica filtros para unir a matriz de pontos do vidro (RAM/Fiat)
        e reduzir reflexos de luz/insulfilm.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contraste = clahe.apply(gray)
        suavizado = cv2.bilateralFilter(contraste, d=9, sigmaColor=75, sigmaSpace=75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.dilate(suavizado, kernel, iterations=1)

    @staticmethod
    def normalizar_chassi(texto: str) -> str:
        """Remove caracteres inválidos mantendo apenas alfanuméricos em caixa alta."""
        return re.sub(r'[^A-Z0-9]', '', (texto or "").upper())

    @staticmethod
    def calcular_match_chassi(texto_lido: str, chassi_oficial: str) -> Tuple[bool, str, List[str]]:
        """
        Compara o que foi lido com o chassi oficial do sistema.
        No vidro normalmente aparecem só os últimos 6 a 8 caracteres (VIS).
        """
        if not chassi_oficial:
            return False, "Chassi oficial não informado no sistema.", []

        oficial = LocalOCREngine.normalizar_chassi(chassi_oficial)
        lido = LocalOCREngine.normalizar_chassi(texto_lido)

        if not lido:
            return False, "Nenhum caractere detectado na imagem.", []

        ambiguidades = []
        sufixo_6 = oficial[-6:] if len(oficial) >= 6 else oficial
        sufixo_8 = oficial[-8:] if len(oficial) >= 8 else oficial

        if sufixo_8 in lido or sufixo_6 in lido:
            return True, f"Match exato com sufixo do chassi ({sufixo_8}).", []

        # Ambiguidades comuns em vidros (8/B, 0/O, 1/I, 5/S, 2/Z)
        mapa_confusao = {
            '8': 'B', 'B': '8',
            '0': 'O', 'O': '0',
            '1': 'I', 'I': '1',
            '5': 'S', 'S': '5',
            '2': 'Z', 'Z': '2'
        }
        for original, substituto in mapa_confusao.items():
            lido_ajustado = lido.replace(original, substituto)
            if sufixo_8 in lido_ajustado or sufixo_6 in lido_ajustado:
                msg = f"Match com correção de ambiguidade ('{original}' lido como '{substituto}' - típico de RAM/vidros)."
                ambiguidades.append(msg)
                return True, msg, ambiguidades

        return False, f"Chassi lido ({lido[-8:] if len(lido)>=8 else lido}) diverge do sistema ({sufixo_8}).", ambiguidades

    def processar_foto_chassi(self, caminho_imagem: str, chassi_oficial: Optional[str] = None) -> ResultadoOCR:
        """Pipeline completo: leitura da imagem, pré-processamento e extração."""
        img = cv2.imread(caminho_imagem)
        if img is None:
            return ResultadoOCR(texto_bruto="", detalhes_match="Não foi possível abrir a imagem. Conferir manualmente.")

        ocr = self._obter_ocr()
        if ocr is None:
            return ResultadoOCR(
                texto_bruto="",
                detalhes_match=f"OCR indisponível ({self.erro_carregamento}). Conferir o chassi manualmente.",
            )

        img_tratada = self.preprocessar_vidro_pontilhado(img)
        try:
            linhas, confiancas = self._executar_ocr(ocr, img_tratada)
        except Exception as e:
            print(f"[Erro OCR Paddle] {e}")
            return ResultadoOCR(texto_bruto="", detalhes_match=f"Erro no OCR ({e}). Conferir o chassi manualmente.")

        texto_extraido = " ".join(linhas)
        confianca_media = float(np.mean(confiancas)) if confiancas else 0.0

        if not texto_extraido.strip():
            return ResultadoOCR(texto_bruto="", detalhes_match="Nenhum texto lido na foto. Conferir o chassi manualmente.")

        match_ok, detalhes, ambiguidades = self.calcular_match_chassi(texto_extraido, chassi_oficial or "")

        return ResultadoOCR(
            texto_bruto=texto_extraido,
            chassi_extraido=self.normalizar_chassi(texto_extraido),
            confianca=round(confianca_media, 2),
            match_com_cadastro=match_ok,
            detalhes_match=detalhes,
            ambiguidades_detectadas=ambiguidades
        )


ocr_engine = LocalOCREngine()
