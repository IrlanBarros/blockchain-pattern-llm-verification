"""Checkpointing stateful para retomada de execução após interrupção.

Feature 1 — Stateful Checkpointing: persiste o progresso do pipeline em JSON
após cada item processado. Se o limite diário de tokens da API for esgotado, o
usuário pode retomar o script no dia seguinte exatamente do ponto onde parou,
sem reprocessar registros já concluídos.

Uso em stages.py:

    cp = Checkpoint(run_dir / "stage1_checkpoint.json", stage="stage1", run_id=run_id)
    for item in prepared_issues:
        if cp.is_completed(item.custom_id_stage1):
            continue          # já processado — pula chamada à API
        try:
            # ... chamar API, salvar resultado ...
            cp.mark_completed(item.custom_id_stage1)
        except Exception:
            cp.mark_failed(item.custom_id_stage1)
            raise

Ao retomar:
    - Itens com ``custom_id`` em ``completed_ids`` são ignorados.
    - Seus resultados anteriores são recarregados do JSONL de raw responses.
    - Itens com ``custom_id`` em ``failed_ids`` são retentados normalmente.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import utc_now_iso, write_json


@dataclass
class CheckpointState:
    """Estado serializável de checkpoint de um stage do pipeline."""

    stage: str
    run_id: str
    completed_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "run_id": self.run_id,
            "completed_ids": self.completed_ids,
            "failed_ids": self.failed_ids,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        return cls(
            stage=data.get("stage", ""),
            run_id=data.get("run_id", ""),
            completed_ids=list(data.get("completed_ids", [])),
            failed_ids=list(data.get("failed_ids", [])),
            last_updated=data.get("last_updated", ""),
        )


class Checkpoint:
    """Gerenciador de checkpoints com persistência automática em JSON.

    Cria um novo checkpoint se o arquivo não existir; caso contrário, carrega
    o estado anterior e exibe um resumo do progresso anterior.

    Args:
        path: Caminho do arquivo de checkpoint (``stage1_checkpoint.json``).
        stage: Nome do stage (``"stage1"`` ou ``"stage2"``).
        run_id: Identificador da execução atual.
    """

    def __init__(self, path: Path, *, stage: str, run_id: str) -> None:
        self.path = path
        self._state = self._load_or_create(stage, run_id)

    # ------------------------------------------------------------------
    # Carregamento / criação
    # ------------------------------------------------------------------

    def _load_or_create(self, stage: str, run_id: str) -> CheckpointState:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                state = CheckpointState.from_dict(data)
                print(
                    f"[checkpoint] Retomando {stage}: "
                    f"{len(state.completed_ids)} concluídos, "
                    f"{len(state.failed_ids)} com falha anteriores."
                )
                return state
            except Exception as exc:
                print(
                    f"[checkpoint] Aviso: falha ao carregar {self.path}: {exc}. "
                    "Iniciando novo checkpoint."
                )
        return CheckpointState(stage=stage, run_id=run_id)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def is_completed(self, custom_id: str) -> bool:
        """Retorna True se *custom_id* já foi concluído com sucesso."""
        return custom_id in self._state.completed_ids

    def mark_completed(self, custom_id: str) -> None:
        """Registra *custom_id* como concluído e persiste imediatamente."""
        if custom_id not in self._state.completed_ids:
            self._state.completed_ids.append(custom_id)
        self._state.last_updated = utc_now_iso()
        self._persist()

    def mark_failed(self, custom_id: str) -> None:
        """Registra *custom_id* como falho e persiste imediatamente."""
        if custom_id not in self._state.failed_ids:
            self._state.failed_ids.append(custom_id)
        self._state.last_updated = utc_now_iso()
        self._persist()

    def completed_count(self) -> int:
        return len(self._state.completed_ids)

    def failed_count(self) -> int:
        return len(self._state.failed_ids)

    def _persist(self) -> None:
        write_json(self.path, self._state.to_dict())
