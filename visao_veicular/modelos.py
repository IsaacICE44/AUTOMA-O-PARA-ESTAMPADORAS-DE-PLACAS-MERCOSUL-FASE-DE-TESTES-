from pydantic import BaseModel, Field


class ResultadoOCR(BaseModel):
    texto_bruto: str = ""
    chassi_extraido: str | None = None
    confianca: float = 0.0
    match_com_cadastro: bool = False
    detalhes_match: str = ""
    ambiguidades_detectadas: list[str] = Field(default_factory=list)


class ResultadoCor(BaseModel):
    cor_detectada: str = "DESCONHECIDA"
    confianca: float = 0.0
    match_com_cadastro: bool = False