"""Deterministic prompt-contract regressions from the 200-case human review.

These tests do not pretend to predict a remote model. They verify that every
reviewed failure mode is represented in the runtime prompt, that implicit
mechanisms reach the model with their catalog definitions, and that mocked
yes/no responses preserve the existing JSON/parser contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_pipeline.data import load_patterns, prepare_issue
from llm_pipeline.lexical import detect_false_friend_signals
from llm_pipeline.prompts import STAGE1_RULES, STAGE2_RULES
from llm_pipeline.requests import stage1_request_params, stage2_request_params
from llm_pipeline.schemas import normalize_stage2, stage2_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog():
    return load_patterns(PROJECT_ROOT / "blockchain_patterns_keywords_v3.csv")


def prepared(text: str):
    return prepare_issue(
        pd.Series(
            {
                "repository": "review/regression",
                "issue_number": "1",
                "issue_title": "Semantic regression fixture",
                "issue_body": text,
                "concatenated_comments": "",
                "type": "Issue",
                "labels": "",
                "state": "open",
            }
        ),
        12_000,
    )


CASES = [
    pytest.param(
        "Factory contract",
        "Use CREATE2 to calculate the deterministic address before deployment; this module never creates contract instances.",
        "no",
        id="factory-create2-address-only-no",
    ),
    pytest.param(
        "Factory contract",
        "The deployment manager contract creates and deploys a new vault contract instance for every market.",
        "yes",
        id="factory-creates-contracts-yes",
    ),
    pytest.param(
        "Append-only data",
        "Append entries to the list, but administrators may later remove, overwrite, or delete any entry.",
        "no",
        id="append-mutable-no",
    ),
    pytest.param(
        "Append-only data",
        "Add new audit records only; prior records are immutable and cannot be overwritten or removed, preserving history.",
        "yes",
        id="append-history-invariant-yes",
    ),
    pytest.param(
        "Off-chain computation",
        "Compute the UI table locally in the browser before rendering; no contract consumes or verifies the result.",
        "no",
        id="local-ui-calculation-no",
    ),
    pytest.param(
        "Off-chain computation",
        "Move the expensive calculation out of the EVM, submit its result to the contract, and verify the proof on-chain.",
        "yes",
        id="moved-computation-result-verified-yes",
    ),
    pytest.param(
        "Low contract footprint",
        "Compress P2P network packets and reduce gossip message payload size between nodes.",
        "no",
        id="network-packets-no",
    ),
    pytest.param(
        "Low contract footprint",
        "Reduce SSTORE writes and contract storage slots to lower gas consumed by the on-chain operation.",
        "yes",
        id="contract-storage-gas-yes",
    ),
    pytest.param(
        "Tokenization",
        "Add an existing ERC-20 address to the allowlist and update its display metadata and icon.",
        "no",
        id="existing-token-listing-no",
    ),
    pytest.param(
        "Tokenization",
        "Represent each real-estate ownership share as an on-chain transferable token and issue it to the owner.",
        "yes",
        id="asset-representation-yes",
    ),
    pytest.param(
        "Proxy contract",
        "Preserve storage while delegatecall forwards to an implementation address whose target can be changed for upgrades.",
        "yes",
        id="implicit-proxy-mechanism-yes",
    ),
    pytest.param(
        "Proxy contract",
        "The system happens to use a proxy contract, but this issue only fixes CSS spacing on the settings button.",
        "no",
        id="proxy-background-no",
    ),
    pytest.param(
        "Reverse Oracle",
        "An off-chain worker subscribes to contract events and reacts to the emitted on-chain state by updating the backend.",
        "yes",
        id="implicit-reverse-oracle-yes",
    ),
    pytest.param(
        "Reverse Oracle",
        "A blockchain event listener is already deployed; this issue only changes the unrelated email template colors.",
        "no",
        id="reverse-oracle-background-no",
    ),
    pytest.param(
        "Oracle",
        "An external service fetches the real-world exchange rate and submits that value in a transaction to the smart contract.",
        "yes",
        id="implicit-inbound-oracle-yes",
    ),
    pytest.param(
        "Oracle",
        "Upgrade Oracle Database server indexes used by the web application; no blockchain data bridge is involved.",
        "no",
        id="oracle-database-false-friend-no",
    ),
]


def mocked_payload(verdict: str, evidence: str) -> dict[str, object]:
    positive = verdict == "yes"
    return {
        "verdict": verdict,
        "mechanism_match": positive,
        "scope_match": positive,
        "focus_match": positive,
        "evidence_text": evidence,
        "evidence_location": ["issue_body"],
        "justification": "The mocked verdict follows the reviewed mechanism boundary.",
        "adoption_status": "conceptual_discussion" if positive else "not_related",
        "false_friend_detected": "no",
        "pattern_challenge_categories": ["none_explicit"] if positive else [],
        "confidence": "high",
        "alternative_pattern": "",
        "overlap_with": [],
    }


@pytest.mark.parametrize("pattern,text,expected", CASES)
def test_review_scenarios_reach_runtime_prompt_and_keep_schema(catalog, pattern, text, expected):
    item = prepared(text)
    params = stage2_request_params(
        item, pattern, catalog, "mock-model", 0.0, 2048, 42, "low"
    )
    system_instruction = params["config"]["system_instruction"]
    user_text = params["contents"][0]["parts"][0]["text"]

    assert STAGE2_RULES in system_instruction
    assert f"CANDIDATE PATTERN: {pattern}" in user_text
    assert f"FULL DESCRIPTION: {catalog.by_name[pattern]['description']}" in user_text
    assert text in user_text

    normalized = normalize_stage2(mocked_payload(expected, text), catalog)
    assert normalized["verdict"] == expected


@pytest.mark.parametrize(
    "required_rule",
    [
        "enabling primitive or deployment/addressing technique",
        "generic append/add operation",
        "deliberate displacement of blockchain computation",
        "Network-message or packet optimization",
        "already existing token",
        "canonical pattern name is neither necessary nor sufficient",
        "Background: the mechanism exists in the system",
        "SEMANTIC OVERREACH IS DIFFERENT FROM A FALSE FRIEND",
    ],
)
def test_stage2_prompt_contains_generalized_review_boundaries(required_rule):
    combined = STAGE2_RULES
    if required_rule.startswith("SEMANTIC"):
        from llm_pipeline.prompts import COMMON_METHOD_RULES

        combined += COMMON_METHOD_RULES
    assert " ".join(required_rule.split()) in " ".join(combined.split())


@pytest.mark.parametrize(
    "pattern,text",
    [
        (
            "Proxy contract",
            "delegatecall forwards to a replaceable implementation while the public address and storage persist",
        ),
        (
            "Oracle",
            "an off-chain service fetches external weather data and submits it to the smart contract",
        ),
        (
            "Reverse Oracle",
            "a backend worker subscribes to contract events and reacts to on-chain state changes",
        ),
    ],
)
def test_stage1_keeps_implicit_mechanisms_eligible_for_recall(catalog, pattern, text):
    assert pattern.casefold() not in text.casefold()
    params = stage1_request_params(
        prepared(text), catalog, "mock-model", 0.0, 2048, 42, "minimal"
    )
    instruction = params["config"]["system_instruction"]
    assert "canonical or academic pattern name does not\n  need to occur" in instruction
    assert f"- {pattern} [" in instruction
    assert catalog.by_name[pattern]["description"][:120] in instruction


def test_stage2_json_schema_shape_is_unchanged(catalog):
    properties = stage2_schema(catalog.names)["properties"]
    assert set(properties) == {
        "verdict",
        "mechanism_match",
        "scope_match",
        "focus_match",
        "evidence_text",
        "evidence_location",
        "justification",
        "adoption_status",
        "false_friend_detected",
        "pattern_challenge_categories",
        "confidence",
        "alternative_pattern",
        "overlap_with",
    }


@pytest.mark.parametrize(
    "pattern,text,reason",
    [
        ("Vote", "The GitHub community votes and upvotes the feature request.", "forum_or_reaction_vote"),
        ("Snapshotting", "Add an SDK event checkpointer for resume.", "operational_event_checkpoint"),
        ("Mutex", "Use a database lock around the worker thread.", "off_chain_concurrency_lock"),
        ("Off-chain Signatures", "Fix the SignatureRequest component UI.", "programming_or_ui_signature"),
    ],
)
def test_additional_false_friend_signals_are_informative(pattern, text, reason):
    signals = detect_false_friend_signals(text, [pattern])
    assert [(signal.pattern, signal.reason) for signal in signals] == [(pattern, reason)]
