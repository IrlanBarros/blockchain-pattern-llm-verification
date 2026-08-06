"""JSON Schemas e validação semântica das respostas estruturadas."""

from __future__ import annotations

from typing import Any, Iterable

from .config import (
    ADOPTION_STATUSES,
    CHALLENGE_CATEGORIES,
    CONFIDENCE_LEVELS,
    CONTEXT_STATUSES,
    EVIDENCE_LOCATIONS,
    FALSE_FRIEND_VALUES,
    ISSUE_ACTIVITY_TYPES,
    VERDICTS,
)
from .models import PatternCatalog
from .normalization import CanonicalLookup, clean_text, clean_token


def stage1_schema(pattern_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Triagem de uma issue/PR e geração de candidatos a blockchain design patterns.",
        "properties": {
            "issue_summary": {
                "type": "string",
                "description": "Resumo técnico curto do tema real do artefato, sem inferir código ausente.",
            },
            "issue_activity_type": {
                "type": "string",
                "enum": ISSUE_ACTIVITY_TYPES,
                "description": "Tipo principal de atividade do próprio artefato.",
            },
            "issue_challenge_categories": {
                "type": "array",
                "description": "Desafios explicitamente discutidos no nível da issue; use none_explicit quando não houver.",
                "items": {"type": "string", "enum": CHALLENGE_CATEGORIES},
            },
            "context_status": {
                "type": "string",
                "enum": CONTEXT_STATUSES,
                "description": "Se o texto disponível é suficiente para a triagem.",
            },
            "candidates": {
                "type": "array",
                "description": "Zero ou mais patterns com evidência positiva parcial que merecem verificação no Stage 2.",
                "items": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "enum": pattern_names,
                            "description": "Nome canônico exato do catálogo.",
                        },
                        "evidence_text": {
                            "type": "string",
                            "description": "Trecho literal curto que sustenta a candidatura; não use somente a keyword.",
                        },
                        "evidence_location": {
                            "type": "string",
                            "enum": EVIDENCE_LOCATIONS,
                            "description": "Local do trecho literal no artefato.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Uma frase ligando a evidência ao mecanismo distintivo do pattern.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_LEVELS,
                            "description": "Confiança de que o par merece o Stage 2, não o veredito final.",
                        },
                    },
                    "required": [
                        "pattern",
                        "evidence_text",
                        "evidence_location",
                        "rationale",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "issue_summary",
            "issue_activity_type",
            "issue_challenge_categories",
            "context_status",
            "candidates",
        ],
        "additionalProperties": False,
    }


def stage2_schema(pattern_names: list[str]) -> dict[str, Any]:
    optional_pattern_names = [""] + pattern_names
    return {
        "type": "object",
        "description": "Verificação rigorosa de um único par issue-pattern.",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": VERDICTS,
                "description": "yes, no, uncertain ou insufficient_context conforme o manual.",
            },
            "evidence_text": {
                "type": "string",
                "description": "Trecho literal curto; obrigatório para yes/uncertain e útil para explicar negativos.",
            },
            "evidence_location": {
                "type": "array",
                "description": "Um ou mais locais onde a evidência aparece.",
                "items": {"type": "string", "enum": EVIDENCE_LOCATIONS},
            },
            "justification": {
                "type": "string",
                "description": "Uma frase explicando a ligação ou incompatibilidade entre texto e pattern.",
            },
            "adoption_status": {
                "type": "string",
                "enum": ADOPTION_STATUSES,
                "description": "Status textual do pattern no artefato, sem inferir a arquitetura do repositório.",
            },
            "false_friend_detected": {
                "type": "string",
                "enum": FALSE_FRIEND_VALUES,
                "description": "yes somente para colisão lexical direta com outro significado.",
            },
            "pattern_challenge_categories": {
                "type": "array",
                "description": "Desafios explicitamente conectados ao mecanismo deste pattern.",
                "items": {"type": "string", "enum": CHALLENGE_CATEGORIES},
            },
            "confidence": {
                "type": "string",
                "enum": CONFIDENCE_LEVELS,
                "description": "Confiança no veredito do par.",
            },
            "alternative_pattern": {
                "type": "string",
                "enum": optional_pattern_names,
                "description": "Nome canônico de pattern alternativo mais adequado, ou string vazia.",
            },
            "overlap_with": {
                "type": "array",
                "description": "Patterns conceitualmente sobrepostos à mesma evidência; não implica positividade.",
                "items": {"type": "string", "enum": pattern_names},
            },
        },
        "required": [
            "verdict",
            "evidence_text",
            "evidence_location",
            "justification",
            "adoption_status",
            "false_friend_detected",
            "pattern_challenge_categories",
            "confidence",
            "alternative_pattern",
            "overlap_with",
        ],
        "additionalProperties": False,
    }


def normalize_enum(value: str, allowed: Iterable[str], field: str) -> str:
    return CanonicalLookup(allowed, label=field).canonicalize(value)


def normalize_list(values: Any, allowed: Iterable[str], field: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field} deve ser lista; recebido {type(values).__name__}")
    lookup = CanonicalLookup(allowed, label=field)
    result: list[str] = []
    for value in values:
        canonical = lookup.canonicalize(value)
        if canonical not in result:
            result.append(canonical)
    return result


def _normalize_challenges(values: Any, field: str, *, default_none: bool) -> list[str]:
    result = normalize_list(values, CHALLENGE_CATEGORIES, field)
    if "none_explicit" in result and len(result) > 1:
        result = [value for value in result if value != "none_explicit"]
    if not result and default_none:
        result = ["none_explicit"]
    return result


def normalize_stage1(payload: dict[str, Any], catalog: PatternCatalog) -> dict[str, Any]:
    result = {
        "issue_summary": clean_text(payload.get("issue_summary")),
        "issue_activity_type": normalize_enum(
            clean_token(payload.get("issue_activity_type")), ISSUE_ACTIVITY_TYPES, "issue_activity_type"
        ),
        "issue_challenge_categories": _normalize_challenges(
            payload.get("issue_challenge_categories"),
            "issue_challenge_categories",
            default_none=True,
        ),
        "context_status": normalize_enum(
            clean_token(payload.get("context_status")), CONTEXT_STATUSES, "context_status"
        ),
        "candidates": [],
    }
    if not result["issue_summary"]:
        raise ValueError("issue_summary não pode ser vazio")

    seen: set[str] = set()
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("candidates deve ser lista")
    pattern_lookup = CanonicalLookup(catalog.names, label="pattern_catalog")
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("Cada candidato deve ser objeto")
        pattern = pattern_lookup.canonicalize(item.get("pattern"))
        if pattern in seen:
            continue
        seen.add(pattern)
        evidence = clean_text(item.get("evidence_text"))
        rationale = clean_text(item.get("rationale"))
        if not evidence:
            raise ValueError(f"Candidato {pattern} sem evidence_text")
        if not rationale:
            raise ValueError(f"Candidato {pattern} sem rationale")
        result["candidates"].append(
            {
                "pattern": pattern,
                "evidence_text": evidence,
                "evidence_location": normalize_enum(
                    clean_token(item.get("evidence_location")), EVIDENCE_LOCATIONS, "evidence_location"
                ),
                "rationale": rationale,
                "confidence": normalize_enum(
                    clean_token(item.get("confidence")), CONFIDENCE_LEVELS, "confidence"
                ),
            }
        )
    return result


def normalize_stage2(payload: dict[str, Any], catalog: PatternCatalog) -> dict[str, Any]:
    pattern_lookup = CanonicalLookup(catalog.names, label="pattern_catalog")
    alternative = pattern_lookup.canonicalize(payload.get("alternative_pattern"), allow_blank=True)

    overlaps = payload.get("overlap_with", [])
    if not isinstance(overlaps, list):
        raise ValueError("overlap_with deve ser lista")
    normalized_overlaps: list[str] = []
    for item in overlaps:
        name = pattern_lookup.canonicalize(item)
        if name not in normalized_overlaps:
            normalized_overlaps.append(name)

    result = {
        "verdict": normalize_enum(clean_token(payload.get("verdict")), VERDICTS, "verdict"),
        "evidence_text": clean_text(payload.get("evidence_text")),
        "evidence_location": normalize_list(
            payload.get("evidence_location"), EVIDENCE_LOCATIONS, "evidence_location"
        ),
        "justification": clean_text(payload.get("justification")),
        "adoption_status": normalize_enum(
            clean_token(payload.get("adoption_status")), ADOPTION_STATUSES, "adoption_status"
        ),
        "false_friend_detected": normalize_enum(
            clean_token(payload.get("false_friend_detected")),
            FALSE_FRIEND_VALUES,
            "false_friend_detected",
        ),
        "pattern_challenge_categories": _normalize_challenges(
            payload.get("pattern_challenge_categories"),
            "pattern_challenge_categories",
            default_none=False,
        ),
        "confidence": normalize_enum(
            clean_token(payload.get("confidence")), CONFIDENCE_LEVELS, "confidence"
        ),
        "alternative_pattern": alternative,
        "overlap_with": normalized_overlaps,
    }

    verdict = result["verdict"]
    status = result["adoption_status"]
    if not result["justification"]:
        raise ValueError("justification não pode ser vazia")
    if verdict == "yes" and status in {"superficial_mention", "not_related", "insufficient_context"}:
        raise ValueError(f"Inconsistência: verdict=yes com adoption_status={status}")
    if verdict == "no" and status not in {"superficial_mention", "not_related"}:
        raise ValueError(f"Inconsistência: verdict=no com adoption_status={status}")
    if verdict == "insufficient_context" and status != "insufficient_context":
        raise ValueError("Inconsistência: insufficient_context exige adoption_status=insufficient_context")
    if verdict == "uncertain" and status in {"not_related", "superficial_mention"}:
        raise ValueError(f"Inconsistência: verdict=uncertain com adoption_status={status}")
    if verdict in {"yes", "uncertain"} and not result["evidence_text"]:
        raise ValueError(f"{verdict} exige evidence_text")
    if verdict in {"yes", "uncertain"} and not result["evidence_location"]:
        raise ValueError(f"{verdict} exige evidence_location")
    if verdict == "no":
        # Não associa desafio ao pattern ausente.
        result["pattern_challenge_categories"] = []
    elif verdict in {"yes", "uncertain"} and not result["pattern_challenge_categories"]:
        result["pattern_challenge_categories"] = ["none_explicit"]
    return result
