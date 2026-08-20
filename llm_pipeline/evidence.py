"""Validação de evidências textuais: somente substrings literais ou com whitespace normalizado.

P5 — Evidence Alteration fix: evidence_text deve ser um trecho literal extraído do artefato.
Aceita correspondência exata ou com espaçamento interno normalizado (colapso de múltiplos
espaços/quebras de linha em espaço único), mas não aceita reescrita ou paráfrase.
"""

from __future__ import annotations

import re


def _normalize_ws(text: str) -> str:
    """Colapsa sequências de whitespace em um único espaço e remove margens."""
    return re.sub(r"\s+", " ", text).strip()


def is_literal_match(evidence: str, artifact_text: str) -> bool:
    """Retorna True se *evidence* é substring literal (ou com whitespace normalizado) de *artifact_text*.

    Dois critérios em ordem de preferência:
    1. Correspondência exata: ``evidence in artifact_text``.
    2. Correspondência com whitespace normalizado: ambos os textos têm sequências de espaços
       colapsadas em um único espaço antes da comparação.

    Retorna False quando *evidence* ou *artifact_text* é vazio.
    """
    if not evidence or not artifact_text:
        return False
    if evidence in artifact_text:
        return True
    return _normalize_ws(evidence) in _normalize_ws(artifact_text)
