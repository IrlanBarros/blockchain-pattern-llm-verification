"""Rastreamento de estabilidade e reprodutibilidade da execução do pipeline.

P11 — Execution Consistency fix: registra hashes dos parâmetros de configuração,
schemas e artefatos de entrada para permitir comparação entre execuções e detectar
divergências de configuração. Documenta explicitamente o não-determinismo residual
inerente ao LLM.

Nota: reprodutibilidade byte-a-byte é garantida apenas nas etapas determinísticas
(normalização, truncamento, validação de schema). Saídas do LLM são inerentemente
não determinísticas mesmo com temperature=0 e seed fixo, devido a diferenças de
infraestrutura do servidor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import utc_now_iso, write_json

_LLM_NONDETERMINISM_NOTE = (
    "Residual LLM non-determinism: even with temperature=0 and a fixed seed, "
    "the Gemini API may produce slightly different outputs across calls due to "
    "server-side infrastructure differences. This is expected and cannot be "
    "eliminated at the client level."
)

_DETERMINISTIC_STAGES_NOTE = (
    "Byte-for-byte reproducibility is guaranteed only for deterministic preprocessing "
    "stages: normalization, truncation, schema validation, and aggregation. "
    "LLM call outputs are excluded from this guarantee."
)


@dataclass
class StabilityReport:
    """Relatório de reprodutibilidade de uma execução do pipeline."""

    run_id: str
    parameter_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def record_param(self, key: str, value: Any) -> None:
        """Registra o hash SHA-256 (primeiros 16 hex) do valor serializado."""
        serialized = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        self.parameter_hashes[key] = hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def add_note(self, note: str) -> None:
        """Adiciona uma nota descritiva ao relatório."""
        self.notes.append(note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "parameter_hashes": self.parameter_hashes,
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        """Persiste o relatório em JSON."""
        write_json(path, self.to_dict())


def build_stability_report(
    run_id: str,
    args: Any,
    schema_hashes: dict[str, str],
) -> StabilityReport:
    """Constrói um :class:`StabilityReport` completo para uma execução.

    Args:
        run_id: Identificador único da execução.
        args: Namespace de argumentos CLI (argparse.Namespace).
        schema_hashes: Dicionário ``{campo: sha256_hex}`` dos schemas Stage 1 e Stage 2.

    Returns:
        Instância populada de :class:`StabilityReport`.
    """
    report = StabilityReport(run_id=run_id)

    # Parâmetros que afetam diretamente o comportamento do LLM
    for attr in (
        "temperature",
        "seed",
        "stage1_model",
        "stage2_model",
        "stage1_thinking_level",
        "stage2_thinking_level",
        "max_input_chars",
        "stage1_max_tokens",
        "stage2_max_tokens",
    ):
        report.record_param(attr, getattr(args, attr, None))

    # Hashes de schema e prompts
    for key, value in schema_hashes.items():
        report.record_param(key, value)

    report.add_note(_LLM_NONDETERMINISM_NOTE)
    report.add_note(_DETERMINISTIC_STAGES_NOTE)
    return report
