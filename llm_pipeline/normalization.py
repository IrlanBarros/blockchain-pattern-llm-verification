"""Normalização estrutural auditável de CSVs, campos e nomes canônicos.

A normalização deste módulo é deliberadamente conservadora: corrige apenas
variações de codificação, Unicode, espaços, separadores e capitalização. Ela
nunca troca uma decisão semântica nem aproxima um nome desconhecido de um
pattern válido.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


_DASHES = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",  # minus sign
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _preview(value: Any, limit: int = 160) -> str:
    text = "" if value is None else str(value)
    escaped = text.encode("unicode_escape", errors="backslashreplace").decode("ascii")
    return escaped if len(escaped) <= limit else escaped[: limit - 3] + "..."


def _value_hash(value: Any) -> str:
    return hashlib.sha256(("" if value is None else str(value)).encode("utf-8")).hexdigest()


@dataclass
class NormalizationReport:
    """Rastro das correções estruturais aplicadas a um artefato."""

    source: str
    encoding: str = ""
    delimiter: str = ""
    changes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_change(
        self,
        *,
        scope: str,
        before: Any,
        after: Any,
        reason: str,
        row: int | str | None = None,
        column: str = "",
    ) -> None:
        if str(before) == str(after):
            return
        self.changes.append(
            {
                "source": self.source,
                "scope": scope,
                "row": row,
                "column": column,
                "reason": reason,
                "before_preview": _preview(before),
                "after_preview": _preview(after),
                "before_sha256": _value_hash(before),
                "after_sha256": _value_hash(after),
            }
        )

    def summary(self) -> dict[str, Any]:
        by_reason: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        for item in self.changes:
            by_reason[item["reason"]] = by_reason.get(item["reason"], 0) + 1
            by_scope[item["scope"]] = by_scope.get(item["scope"], 0) + 1
        return {
            "source": self.source,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "change_count": len(self.changes),
            "changes_by_reason": by_reason,
            "changes_by_scope": by_scope,
            "warnings": self.warnings,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "changes": self.changes}


def normalize_unicode(value: Any, *, preserve_newlines: bool = True) -> str:
    """Aplica NFKC, remove invisíveis e normaliza separadores Unicode.

    Em texto livre, quebras de linha e tabs são preservadas. Em tokens e
    categorias, elas são posteriormente compactadas por :func:`clean_token`.
    """

    if _is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for source, target in _DASHES.items():
        text = text.replace(source, target)

    out: list[str] = []
    for char in text:
        if char == "\n":
            if preserve_newlines:
                out.append(char)
            else:
                out.append(" ")
            continue
        if char == "\t":
            out.append(char if preserve_newlines else " ")
            continue

        category = unicodedata.category(char)
        if category == "Cf":
            # P10: Preserve characters essential for compound sequences.
            # U+200D ZERO WIDTH JOINER (ZWJ) is used in multi-codepoint emoji (e.g. 👨\u200d💻).
            # U+200C ZERO WIDTH NON-JOINER (ZWNJ) is used in Indic/Persian scripts.
            # All other Cf characters (BOM, zero-width spaces, soft hyphens…) are stripped.
            if char in ("\u200d", "\u200c"):
                out.append(char)
            continue
        if category == "Cc":
            # Demais caracteres de controle não têm função útil em CSVs.
            continue
        if category.startswith("Z"):
            if preserve_newlines and category in {"Zl", "Zp"}:
                out.append("\n")
            else:
                out.append(" ")
            continue
        out.append(char)
    return "".join(out)


def clean_text(value: Any) -> str:
    """Limpa texto livre sem destruir espaçamento interno ou blocos de código."""

    return normalize_unicode(value, preserve_newlines=True).strip()


def clean_token(value: Any) -> str:
    """Limpa identificadores/categorias e compacta qualquer whitespace."""

    text = normalize_unicode(value, preserve_newlines=False)
    return re.sub(r"\s+", " ", text).strip()


def canonical_display(value: Any) -> str:
    """Normaliza pontuação estrutural de um nome, sem trocar suas palavras."""

    text = clean_token(value)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def structural_fingerprint(value: Any) -> str:
    """Fingerprint que ignora apenas caixa, whitespace e pontuação/separadores.

    Assim, ``Role Based Control``, ``role_based_control`` e
    ``Role-based control`` são equivalentes. Sinônimos ou palavras diferentes
    continuam diferentes.
    """

    text = canonical_display(value).casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_header(value: Any) -> str:
    text = clean_token(value).casefold()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_")


class CanonicalLookup:
    """Resolve variantes estruturais para um conjunto fechado de valores."""

    def __init__(self, allowed: Iterable[str], *, label: str):
        self.label = label
        self.allowed = [canonical_display(item) for item in allowed]
        self._exact = {item: item for item in self.allowed}
        self._fingerprint: dict[str, str] = {}
        collisions: dict[str, list[str]] = {}
        for item in self.allowed:
            fp = structural_fingerprint(item)
            previous = self._fingerprint.get(fp)
            if previous is not None and previous != item:
                collisions.setdefault(fp, [previous]).append(item)
            self._fingerprint[fp] = item
        if collisions:
            raise ValueError(
                f"Valores ambíguos em {label} após normalização estrutural: {collisions}"
            )

    def canonicalize(
        self,
        value: Any,
        *,
        allow_blank: bool = False,
        report: NormalizationReport | None = None,
        row: int | str | None = None,
        column: str = "",
    ) -> str:
        original = "" if _is_missing(value) else str(value)
        cleaned = canonical_display(value)
        if not cleaned:
            if allow_blank:
                return ""
            raise ValueError(f"Valor vazio em {self.label}")
        if cleaned in self._exact:
            canonical = self._exact[cleaned]
        else:
            fp = structural_fingerprint(cleaned)
            canonical = self._fingerprint.get(fp, "")
            if not canonical:
                raise ValueError(
                    f"Valor fora de {self.label}: {original!r}. "
                    "A normalização não adivinha sinônimos nem decisões semânticas."
                )
        if report is not None and original != canonical:
            report.add_change(
                scope="canonical_value",
                before=original,
                after=canonical,
                reason=f"canonicalized_{self.label}",
                row=row,
                column=column,
            )
        return canonical


def split_multivalue(value: Any) -> list[str]:
    """Lê lista JSON ou valores separados por ``|``, removendo vazios."""

    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = clean_text(value)
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw = parsed
                else:
                    raw = [text]
            except json.JSONDecodeError:
                raw = text.split("|")
        else:
            raw = text.split("|")
    result: list[str] = []
    for item in raw:
        token = clean_token(item)
        if token and token not in result:
            result.append(token)
    return result


def canonicalize_multivalue(
    value: Any,
    lookup: CanonicalLookup,
    *,
    report: NormalizationReport | None = None,
    row: int | str | None = None,
    column: str = "",
) -> list[str]:
    result: list[str] = []
    for item in split_multivalue(value):
        canonical = lookup.canonicalize(
            item, report=report, row=row, column=column
        )
        if canonical not in result:
            result.append(canonical)
    return result


def normalize_dataframe_columns(
    df: pd.DataFrame,
    report: NormalizationReport | None = None,
) -> pd.DataFrame:
    """Normaliza cabeçalhos e rejeita colisões em vez de sobrescrever dados."""

    mapping: dict[Any, str] = {}
    reverse: dict[str, list[str]] = {}
    for original in df.columns:
        canonical = normalize_header(original)
        if not canonical:
            raise ValueError(f"Cabeçalho vazio/inválido após normalização: {original!r}")
        mapping[original] = canonical
        reverse.setdefault(canonical, []).append(str(original))
        if report is not None and str(original) != canonical:
            report.add_change(
                scope="column",
                before=original,
                after=canonical,
                reason="normalized_column_name",
                column=canonical,
            )
    collisions = {name: values for name, values in reverse.items() if len(values) > 1}
    if collisions:
        raise ValueError(
            "Duas ou mais colunas tornam-se iguais após a normalização; "
            f"corrija o CSV para evitar perda de dados: {collisions}"
        )
    return df.rename(columns=mapping)



def resolve_repository_columns(
    df: pd.DataFrame,
    report: NormalizationReport | None = None,
    *,
    require: bool = True,
) -> pd.DataFrame:
    """Aceita ``repository_full_name`` ou ``repository`` como identificador.

    O corpus e o smoke test usam ``repository_full_name``. Internamente, o
    pipeline mantém também a coluna ``repository`` para preservar a chave
    histórica usada pelos stages e pela avaliação. ``repository_category`` é
    preservada apenas como metadado e não participa da chave.

    Se as duas colunas existirem, elas precisam representar o mesmo valor após
    limpeza estrutural. Divergências são rejeitadas em vez de resolvidas
    silenciosamente.
    """

    out = df.copy()
    has_repository = "repository" in out.columns
    has_full_name = "repository_full_name" in out.columns

    if not has_repository and not has_full_name:
        if require:
            raise ValueError(
                "CSV de entrada sem identificador de repositório: informe "
                "'repository_full_name' (preferido no smoke test) ou 'repository'."
            )
        return out

    if not has_repository:
        out["repository"] = out["repository_full_name"]
        if report is not None:
            report.add_change(
                scope="derived_column",
                before="repository_full_name",
                after="repository",
                reason="derived_repository_alias",
                column="repository",
            )

    if not has_full_name:
        out["repository_full_name"] = out["repository"]
        if report is not None:
            report.add_change(
                scope="derived_column",
                before="repository",
                after="repository_full_name",
                reason="derived_repository_full_name_alias",
                column="repository_full_name",
            )

    repository_values = out["repository"].map(clean_token)
    full_name_values = out["repository_full_name"].map(clean_token)

    conflicts = (repository_values != "") & (full_name_values != "") & (
        repository_values != full_name_values
    )
    if conflicts.any():
        examples = []
        for pos in out.index[conflicts][:10]:
            examples.append(
                {
                    "csv_row": int(pos) + 2 if isinstance(pos, int) else str(pos),
                    "repository": repository_values.loc[pos],
                    "repository_full_name": full_name_values.loc[pos],
                }
            )
        raise ValueError(
            "As colunas 'repository' e 'repository_full_name' divergem após "
            f"normalização. Exemplos: {examples}"
        )

    canonical = repository_values.where(repository_values != "", full_name_values)
    canonical = canonical.where(canonical != "", repository_values)

    if (canonical == "").any():
        rows = [int(i) + 2 if isinstance(i, int) else str(i) for i in out.index[canonical == ""][:10]]
        raise ValueError(f"Há identificador de repositório vazio nas linhas CSV: {rows}")

    for pos in out.index:
        before_repository = out.at[pos, "repository"]
        before_full_name = out.at[pos, "repository_full_name"]
        after = canonical.loc[pos]
        out.at[pos, "repository"] = after
        out.at[pos, "repository_full_name"] = after
        if report is not None:
            if str(before_repository) != str(after):
                report.add_change(
                    scope="cell",
                    before=before_repository,
                    after=after,
                    reason="synchronized_repository_alias",
                    row=int(pos) + 2 if isinstance(pos, int) else str(pos),
                    column="repository",
                )
            if str(before_full_name) != str(after):
                report.add_change(
                    scope="cell",
                    before=before_full_name,
                    after=after,
                    reason="synchronized_repository_full_name_alias",
                    row=int(pos) + 2 if isinstance(pos, int) else str(pos),
                    column="repository_full_name",
                )

    return out

def normalize_dataframe_cells(
    df: pd.DataFrame,
    *,
    token_columns: Sequence[str] = (),
    report: NormalizationReport | None = None,
) -> pd.DataFrame:
    """Remove invisíveis de todas as células e compacta colunas-token."""

    out = df.copy()
    token_set = set(token_columns)
    for column in out.columns:
        cleaner = clean_token if column in token_set else clean_text
        normalized_values: list[str] = []
        for pos, value in enumerate(out[column].tolist()):
            before = "" if _is_missing(value) else str(value)
            after = cleaner(value)
            normalized_values.append(after)
            if report is not None and before != after:
                report.add_change(
                    scope="cell",
                    before=before,
                    after=after,
                    reason=(
                        "normalized_token_whitespace_unicode"
                        if column in token_set
                        else "normalized_text_unicode_boundaries"
                    ),
                    row=pos + 2,
                    column=column,
                )
        out[column] = normalized_values
    return out


def _decode_csv_bytes(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        raise UnicodeError("CSV não pôde ser decodificado como UTF-8/UTF-8-BOM/CP1252.") from None


def _detect_delimiter(text: str) -> str:
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {delimiter: first_line.count(delimiter) for delimiter in (",", ";", "\t")}
        delimiter = max(counts, key=counts.get)
        return delimiter if counts[delimiter] else ","


def read_csv_robust(path: Path, *, source: str | None = None) -> tuple[pd.DataFrame, NormalizationReport]:
    """Lê CSV preservando strings e aceitando BOM, CP1252 e separadores comuns."""

    raw = path.read_bytes()
    text, encoding = _decode_csv_bytes(raw)
    delimiter = _detect_delimiter(text)
    report = NormalizationReport(source=source or str(path), encoding=encoding, delimiter=delimiter)

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        raw_header = next(reader)
    except StopIteration:
        raise ValueError(f"CSV vazio: {path}") from None
    normalized_headers = [normalize_header(item) for item in raw_header]
    collisions: dict[str, list[str]] = {}
    for index, canonical in enumerate(normalized_headers):
        same = [raw_header[i] for i, name in enumerate(normalized_headers) if name == canonical]
        if len(same) > 1:
            collisions[canonical] = same
    if collisions:
        raise ValueError(
            "Cabeçalhos duplicados após normalização, antes da leitura do CSV: "
            f"{collisions}"
        )

    df = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    df = normalize_dataframe_columns(df, report)
    return df, report


def merge_reports(*reports: NormalizationReport) -> dict[str, Any]:
    return {
        "reports": [report.to_dict() for report in reports],
        "total_changes": sum(len(report.changes) for report in reports),
        "warnings": [warning for report in reports for warning in report.warnings],
    }
