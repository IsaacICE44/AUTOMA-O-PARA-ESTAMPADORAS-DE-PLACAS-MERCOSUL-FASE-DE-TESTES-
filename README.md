# Estampadora CBM — Visão Veicular Local

> **Status: em desenvolvimento.** Esta é uma versão experimental. Os resultados de OCR e cor precisam de conferência humana.

Estampadora CBM é uma integração em desenvolvimento para automatizar rotinas de estampadoras de placas veiculares padrão Mercosul. A primeira integração planejada é com o CBM Web, porque o responsável pelo projeto já tem contato com a CBM. O objetivo futuro é disponibilizar a solução para estampadoras em geral, com adaptação aos fluxos de cada operação.

Esta edição reúne os módulos Python de visão da integração. Eles leem marcações de chassi em imagens com PaddleOCR e estimam a cor predominante com OpenCV. As integrações operacionais com CBM Web e WhatsApp fazem parte do projeto principal.

## Instalação

Use Python 3.12. A primeira execução do OCR pode baixar os modelos.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Exemplo com imagem sintética

O exemplo cria uma imagem de teste em uma pasta temporária e passa o arquivo aos dois módulos.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from visao_veicular.color_detector import LocalColorDetector
from visao_veicular.ocr_engine import LocalOCREngine

imagem = np.full((360, 960, 3), (180, 75, 35), dtype=np.uint8)
cv2.putText(imagem, "TESTE OCR", (40, 100), cv2.FONT_HERSHEY_SIMPLEX,
            2, (255, 255, 255), 4)

with TemporaryDirectory() as pasta:
    foto = str(Path(pasta) / "exemplo.png")
    if not cv2.imwrite(foto, imagem):
        raise RuntimeError("Não foi possível gravar a imagem de teste.")
    print(LocalColorDetector().validar_cor(foto))
    print(LocalOCREngine().processar_foto_chassi(foto))
```
## Licença

[MIT](LICENSE).
