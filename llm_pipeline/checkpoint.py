"""Durable, validated checkpoint primitives for the two pipeline stages.

The final stage CSV is the source of truth. The JSON checkpoint is an
audit-friendly progress index updated only after the corresponding CSV row is
atomically persisted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import clean_scalar, stable_custom_id, utc_now_iso, write_dataframe_csv, write_json


class CheckpointIntegrityError(ValueError):
    """A persisted checkpoint/result artifact cannot safely be resumed."""


@dataclass
class CheckpointState:
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
        if not isinstance(data, dict):
            raise CheckpointIntegrityError("checkpoint deve conter um objeto JSON")
        required = {"stage", "run_id", "completed_ids", "failed_ids", "last_updated"}
        missing = required - set(data)
        if missing:
            raise CheckpointIntegrityError(f"checkpoint sem campos obrigatórios: {sorted(missing)}")
        completed, failed = data["completed_ids"], data["failed_ids"]
        if not isinstance(completed, list) or not isinstance(failed, list):
            raise CheckpointIntegrityError("completed_ids e failed_ids devem ser listas")
        if not all(isinstance(value, str) and value for value in completed + failed):
            raise CheckpointIntegrityError("ids do checkpoint devem ser strings não vazias")
        if len(set(completed)) != len(completed) or len(set(failed)) != len(failed):
            raise CheckpointIntegrityError("checkpoint contém ids duplicados")
        if set(completed) & set(failed):
            raise CheckpointIntegrityError("checkpoint contém ids simultaneamente concluídos e falhos")
        return cls(
            stage=data["stage"], run_id=data["run_id"], completed_ids=completed,
            failed_ids=failed, last_updated=data["last_updated"],
        )


class Checkpoint:
    """Progress index persisted atomically after each state transition."""

    def __init__(self, path: Path, *, stage: str, run_id: str) -> None:
        self.path = path
        self._state = self._load_or_create(stage, run_id)

    def _load_or_create(self, stage: str, run_id: str) -> CheckpointState:
        if not self.path.exists():
            return CheckpointState(stage=stage, run_id=run_id)
        try:
            state = CheckpointState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, CheckpointIntegrityError) as exc:
            raise CheckpointIntegrityError(f"checkpoint inválido em {self.path}: {exc}") from exc
        if state.stage != stage or state.run_id != run_id:
            raise CheckpointIntegrityError(
                f"checkpoint {self.path} pertence a stage/run diferente ({state.stage!r}/{state.run_id!r})"
            )
        print(f"[checkpoint] {stage}: {len(state.completed_ids)} concluídos, {len(state.failed_ids)} falhos registrados.")
        return state

    def is_completed(self, custom_id: str) -> bool:
        return custom_id in self._state.completed_ids

    def mark_completed(self, custom_id: str) -> None:
        if custom_id not in self._state.completed_ids:
            self._state.completed_ids.append(custom_id)
        if custom_id in self._state.failed_ids:
            self._state.failed_ids.remove(custom_id)
        self._save()

    def mark_failed(self, custom_id: str) -> None:
        if custom_id in self._state.completed_ids:
            return
        if custom_id not in self._state.failed_ids:
            self._state.failed_ids.append(custom_id)
        self._save()

    def reconcile_completed(self, completed_ids: Iterable[str]) -> None:
        """Bring this index in line with the validated CSV source of truth."""
        valid = list(dict.fromkeys(completed_ids))
        if self._state.completed_ids != valid:
            self._state.completed_ids = valid
            self._state.failed_ids = [value for value in self._state.failed_ids if value not in valid]
            self._save()

    def completed_count(self) -> int:
        return len(self._state.completed_ids)

    def failed_count(self) -> int:
        return len(self._state.failed_ids)

    def _save(self) -> None:
        self._state.last_updated = utc_now_iso()
        write_json(self.path, self._state.to_dict())


class ResultStore:
    """One-row-per-identity durable result store backed by the final CSV.

    Historical CSVs are accepted when they contain the old key fields. A
    missing ``request_status`` is interpreted as a successful legacy row.
    Ambiguous or malformed artifacts fail loudly; no arbitrary deduplication
    occurs during resume.
    """

    def __init__(self, path: Path, *, stage: str) -> None:
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"stage inválido: {stage}")
        self.path, self.stage = path, stage
        self.key_columns = ["repository", "issue_number"] + (["pattern"] if stage == "stage2" else [])
        self.dataframe = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        try:
            dataframe = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise CheckpointIntegrityError(f"não foi possível ler {self.path}: {exc}") from exc
        missing = set(self.key_columns) - set(dataframe.columns)
        if missing:
            raise CheckpointIntegrityError(f"{self.path} sem colunas-chave: {sorted(missing)}")
        dataframe = dataframe.copy()
        for column in self.key_columns:
            dataframe[column] = dataframe[column].map(clean_scalar)
        if (dataframe[self.key_columns] == "").any().any():
            raise CheckpointIntegrityError(f"{self.path} contém identidade vazia")
        if "request_status" not in dataframe.columns:
            dataframe["request_status"] = "succeeded"
        dataframe["request_status"] = dataframe["request_status"].map(clean_scalar)
        duplicated = dataframe.duplicated(self.key_columns, keep=False)
        if duplicated.any():
            examples = dataframe.loc[duplicated, self.key_columns].drop_duplicates().head(10).to_dict("records")
            raise CheckpointIntegrityError(
                f"{self.path} contém identidades duplicadas; corrija/migre o artefato antes de retomar: {examples}"
            )
        successful = dataframe["request_status"] == "succeeded"
        if self.stage == "stage1" and successful.any():
            if "candidates" not in dataframe.columns:
                raise CheckpointIntegrityError(f"{self.path} sem coluna candidates necessária para Stage 1")
            for value in dataframe.loc[successful, "candidates"]:
                try:
                    candidates = json.loads(value or "[]")
                except json.JSONDecodeError as exc:
                    raise CheckpointIntegrityError(f"{self.path} contém candidates inválido: {value!r}") from exc
                if not isinstance(candidates, list) or not all(isinstance(item, str) and item for item in candidates):
                    raise CheckpointIntegrityError(f"{self.path} contém candidates inválido: {value!r}")
        if self.stage == "stage2" and successful.any():
            if "verdict" not in dataframe.columns or (dataframe.loc[successful, "verdict"].map(clean_scalar) == "").any():
                raise CheckpointIntegrityError(f"{self.path} contém resultado Stage 2 concluído sem verdict")
        return dataframe

    def completed_ids(self) -> set[str]:
        if self.dataframe.empty:
            return set()
        rows = self.dataframe.loc[self.dataframe["request_status"] == "succeeded"]
        # Recompute from the composite identity instead of trusting a possibly
        # stale custom_id column left by an older exporter.
        prefix = "s1" if self.stage == "stage1" else "s2"
        return {
            stable_custom_id(prefix, *(clean_scalar(row[column]) for column in self.key_columns))
            for _, row in rows.iterrows()
        }

    def completed_keys(self) -> set[tuple[str, ...]]:
        if self.dataframe.empty:
            return set()
        rows = self.dataframe.loc[self.dataframe["request_status"] == "succeeded"]
        return {tuple(clean_scalar(row[column]) for column in self.key_columns) for _, row in rows.iterrows()}

    def upsert(self, row: dict[str, Any]) -> None:
        key = tuple(clean_scalar(row.get(column)) for column in self.key_columns)
        if not all(key):
            raise CheckpointIntegrityError(f"tentativa de persistir {self.stage} sem identidade completa: {key}")
        new_row = dict(row)
        for column, value in zip(self.key_columns, key):
            new_row[column] = value
        candidate = pd.DataFrame([new_row])
        if self.dataframe.empty:
            self.dataframe = candidate
        else:
            matches = self.dataframe.apply(
                lambda current: tuple(clean_scalar(current[column]) for column in self.key_columns) == key,
                axis=1,
            )
            self.dataframe = pd.concat([self.dataframe.loc[~matches], candidate], ignore_index=True, sort=False)
        write_dataframe_csv(self.path, self.dataframe)

    def ensure_exists(self, columns: list[str]) -> None:
        if not self.path.exists():
            self.dataframe = pd.DataFrame(columns=columns)
            write_dataframe_csv(self.path, self.dataframe)


def resumable_batch(client, *, model, requests, display_name, run_dir, stage, create, wait, poll_seconds):
    """Persist the remote job identity before polling; reuse in-flight jobs on resume.

    A request subset can reuse an earlier superset after some rows were persisted.
    The unavoidable create-before-save network ambiguity is reported, not hidden.
    """
    from .utils import object_to_dict, sha256_text
    path = run_dir / f'{stage}_remote_jobs.json'
    jobs = json.loads(path.read_text()) if path.exists() else []
    fingerprints = {r['metadata']['custom_id']: sha256_text(json.dumps(r, sort_keys=True, ensure_ascii=False))
                    for r in requests}
    selected = None
    for job in reversed(jobs):
        if job['model'] == model and all(job['requests'].get(k) == v for k,v in fingerprints.items()):
            # Failed/expired remote jobs need resubmission. A successful job that
            # returned malformed/missing model results also needs a fresh attempt.
            if job.get('consumed') or job.get('terminal_failure'):
                continue
            selected = job
            break
    if selected is None:
        created = create(client, model=model, requests=requests, display_name=display_name)
        name = clean_scalar(getattr(created, 'name', ''))
        if not name:
            raise ValueError('Batch created without name')
        selected = {'model':model,'name':name,'requests':fingerprints,
                    'request_order':[r['metadata']['custom_id'] for r in requests],
                    'created':object_to_dict(created)}
        jobs.append(selected)
        write_json(path,jobs)
    final = wait(client, selected['name'], poll_seconds)
    created_metadata = {**selected['created'], '_request_order':selected.get('request_order', [])}
    return selected['name'], created_metadata, final


def mark_batch_consumed(run_dir, stage, name):
    path = run_dir / f'{stage}_remote_jobs.json'
    jobs = json.loads(path.read_text())
    for job in jobs:
        if job['name'] == name:
            job['consumed'] = True
    write_json(path, jobs)
