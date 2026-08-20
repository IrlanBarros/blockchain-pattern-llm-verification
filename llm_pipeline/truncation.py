"""Truncamento estrito de artefatos: somente quando o tamanho exceder o limite configurado.

P7 — Erroneous Truncation fix: garante que registros abaixo do limite nunca sejam marcados
como truncados. O sinal booleano `was_truncated` é autoritativo e derivado apenas da diferença
real de comprimento, não de normalização de espaços em branco.
"""

from __future__ import annotations

import math


def head_tail_truncate(text: str, budget: int) -> tuple[str, bool]:
    """Trunca *text* para no máximo *budget* caracteres usando estratégia cabeça+cauda.

    Retorna ``(texto_resultante, foi_truncado)``.

    Garante:
    - Se ``len(text.strip()) <= budget``, retorna ``(text.strip(), False)`` — sem truncamento.
    - Se ``budget <= 80``, aplica corte simples preservando o início.
    - Caso contrário, preserva 65 % do início e 35 % do final, inserindo um marcador central
      com a contagem de caracteres omitidos.
    """
    stripped = text.strip()
    if len(stripped) <= budget:
        return stripped, False
    if budget <= 80:
        return stripped[:budget], True
    marker = f"\n[... {len(stripped) - budget} caracteres omitidos ...]\n"
    usable = max(1, budget - len(marker))
    head = math.ceil(usable * 0.65)
    tail = usable - head
    truncated = stripped[:head].rstrip() + marker + stripped[-tail:].lstrip()
    return truncated, True
