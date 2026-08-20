"""Detecção léxica de falsos-amigos no Stage 1.

P2 — False Friends fix: separa a identificação de colisões lexicais (baseada em regex,
sem custo de API) da validação semântica feita pelo modelo no Stage 2. Ao pré-detectar
candidatos suspeitos em Stage 1, o Stage 2 recebe contexto adicional para aplicar o
critério rigoroso de falso-amigo definido nas regras do manual.

Um sinal de falso-amigo NÃO elimina automaticamente um candidato — ele é informativo
para o modelo e para o pipeline de agregação.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class FalseFriendSignal(NamedTuple):
    """Sinal de colisão léxica detectado para um pattern candidato."""

    pattern: str
    reason: str


# ---------------------------------------------------------------------------
# Mapeamento: (nome_canônico_do_pattern, regex_de_colisão_lexical, motivo)
# Cada entrada cobre casos onde o termo do pattern aparece com OUTRO significado.
# ---------------------------------------------------------------------------
_FALSE_FRIEND_REGEXES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "Proxy contract",
        re.compile(r"\bnginx\s+(?:reverse\s+)?proxy\b|\bhttp\s+proxy\b|\bload.?balancer\b", re.IGNORECASE),
        "nginx_or_http_proxy",
    ),
    (
        "Oracle",
        re.compile(r"\boracle\s+(?:database|corp|db|server|cloud)\b|\bOracle\s+Corp\b", re.IGNORECASE),
        "oracle_database_or_corp",
    ),
    (
        "Snapshotting",
        re.compile(r"\bforge\s+snapshot\b|\btest\s+snapshot\b|\bvm\s+snapshot\b|\bfoundry\s+snapshot\b", re.IGNORECASE),
        "forge_or_test_snapshot",
    ),
    (
        "Relay contract",
        re.compile(r"\brelay\s+chain\b|\bpolkadot\s+relay\b", re.IGNORECASE),
        "relay_chain_parachain",
    ),
    (
        "Blocklist",
        re.compile(r"\bdomain\s+block(?:list)?\b|\bip\s+block(?:list)?\b|\burl\s+block(?:list)?\b", re.IGNORECASE),
        "domain_ip_blocklist",
    ),
    (
        "Router contract",
        re.compile(r"\bip\s+router\b|\bnetwork\s+router\b|\bwifi\s+router\b", re.IGNORECASE),
        "network_router",
    ),
]


def detect_false_friend_signals(
    artifact_text: str,
    candidate_pattern_names: list[str],
) -> list[FalseFriendSignal]:
    """Detecta sinais de colisão léxica no *artifact_text* para os patterns candidatos.

    Só retorna sinais para patterns presentes em *candidate_pattern_names*, evitando
    falsos positivos em patterns não triados pelo Stage 1.

    Args:
        artifact_text: Texto completo do artefato (título + corpo + comentários).
        candidate_pattern_names: Nomes canônicos dos patterns candidatos do Stage 1.

    Returns:
        Lista de :class:`FalseFriendSignal`, possivelmente vazia.
    """
    if not artifact_text or not candidate_pattern_names:
        return []
    pattern_set = set(candidate_pattern_names)
    signals: list[FalseFriendSignal] = []
    for pattern_name, regex, reason in _FALSE_FRIEND_REGEXES:
        if pattern_name not in pattern_set:
            continue
        if regex.search(artifact_text):
            signals.append(FalseFriendSignal(pattern=pattern_name, reason=reason))
    return signals
