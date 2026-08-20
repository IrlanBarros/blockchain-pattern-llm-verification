"""Cálculo determinístico da pontuação de confiança para resultados do Stage 2.

P1 — Excessive Confidence fix: substitui a confiança auto-reportada pelo modelo por um
valor calculado com base em critérios objetivos:

  - Qualidade e comprimento da evidência
  - Correspondência de mecanismo, escopo e foco (campos mechanism_match, scope_match, focus_match)
  - Detecção de falso-amigo
  - Status de adoção (superficial_mention é incompatível com high confidence)
  - Tipo de veredito

A função é determinística — dado o mesmo payload, sempre retorna o mesmo valor — garantindo
reprodutibilidade independente do modelo LLM.
"""

from __future__ import annotations

from typing import Any

# Comprimentos mínimo e forte de evidência (em caracteres)
_MIN_EVIDENCE_LEN = 15
_STRONG_EVIDENCE_LEN = 50


def compute_stage2_confidence(payload: dict[str, Any]) -> str:
    """Calcula ``'high'``, ``'medium'`` ou ``'low'`` para o veredito do Stage 2.

    Regras em ordem de prioridade:

    1. ``insufficient_context`` → sempre ``low``.
    2. Qualquer veredito com falso-amigo confirmado e evidência positiva → ``medium``.
    3. ``no`` com evidência clara → ``medium``; sem evidência → ``low``.
    4. ``uncertain`` com evidência suficiente → ``medium``; caso contrário → ``low``.
    5. ``yes``:
       - falso-amigo detectado → ``low`` (contradição).
       - evidência ausente ou muito curta → ``low``.
       - mechanism_match, scope_match ou focus_match ausente/falso → ``medium``.
       - adoção superficial (superficial_mention) → ``low``.
       - evidência forte (≥ 50 chars) → ``high``; caso contrário → ``medium``.
    """
    verdict: str = payload.get("verdict", "")
    evidence_text: str = payload.get("evidence_text", "") or ""
    false_friend: str = payload.get("false_friend_detected", "no") or "no"
    adoption_status: str = payload.get("adoption_status", "") or ""

    # mechanism/scope/focus default True para compatibilidade com respostas sem esses campos
    mechanism_match: bool = bool(payload.get("mechanism_match", True))
    scope_match: bool = bool(payload.get("scope_match", True))
    focus_match: bool = bool(payload.get("focus_match", True))

    if verdict == "insufficient_context":
        return "low"

    if verdict == "no":
        if false_friend == "yes" and len(evidence_text) >= _MIN_EVIDENCE_LEN:
            return "medium"
        if len(evidence_text) >= _MIN_EVIDENCE_LEN:
            return "medium"
        return "low"

    if verdict == "uncertain":
        if not evidence_text or len(evidence_text) < _MIN_EVIDENCE_LEN:
            return "low"
        if false_friend == "yes":
            return "low"
        return "medium"

    # yes
    if false_friend == "yes":
        return "low"
    if not evidence_text or len(evidence_text) < _MIN_EVIDENCE_LEN:
        return "low"
    if not (mechanism_match and scope_match and focus_match):
        return "medium"
    if adoption_status in {"superficial_mention", "not_related"}:
        return "low"
    if len(evidence_text) >= _STRONG_EVIDENCE_LEN:
        return "high"
    return "medium"
