"""Testes abrangentes para o refactoring P1-P12 e Features 1-2.

Todos os testes usam mocks para não fazer chamadas reais à API Gemini.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_pipeline.checkpoint import Checkpoint, CheckpointState
from llm_pipeline.confidence import compute_stage2_confidence
from llm_pipeline.evidence import is_literal_match
from llm_pipeline.lexical import detect_false_friend_signals
from llm_pipeline.models import PreparedIssue, PatternCatalog
from llm_pipeline.normalization import normalize_unicode
from llm_pipeline.schemas import normalize_stage2
from llm_pipeline.truncation import head_tail_truncate


# ============================================================================
# P7: Truncation tests
# ============================================================================


class TestTruncation:
    """Testes do módulo truncation.py (P7 - Erroneous Truncation fix)."""

    def test_head_tail_truncate_no_truncation_needed(self):
        text = "This is a short text"
        result, was_truncated = head_tail_truncate(text, 100)
        assert result == text.strip()
        assert was_truncated is False

    def test_head_tail_truncate_exactly_at_limit(self):
        text = "x" * 50
        result, was_truncated = head_tail_truncate(text, 50)
        assert len(result) == 50
        assert was_truncated is False

    def test_head_tail_truncate_needed_long_budget(self):
        text = "a" * 200
        result, was_truncated = head_tail_truncate(text, 100)
        assert was_truncated is True
        assert len(result) <= 100
        assert "omitidos" in result

    def test_head_tail_truncate_needed_short_budget(self):
        text = "a" * 200
        result, was_truncated = head_tail_truncate(text, 50)
        assert was_truncated is True
        assert len(result) <= 50

    def test_head_tail_truncate_empty_string(self):
        result, was_truncated = head_tail_truncate("", 50)
        assert result == ""
        assert was_truncated is False

    def test_head_tail_truncate_preserves_content(self):
        text = "Lorem ipsum dolor sit amet. " * 20
        result, was_truncated = head_tail_truncate(text, 150)
        assert was_truncated is True
        assert "Lorem" in result
        assert "amet" in result


# ============================================================================
# P5: Evidence validation tests
# ============================================================================


class TestEvidence:
    """Testes do módulo evidence.py (P5 - Evidence Alteration fix)."""

    def test_is_literal_match_exact(self):
        assert is_literal_match("hello world", "hello world") is True
        assert is_literal_match("hello", "hello world and more") is True

    def test_is_literal_match_whitespace_normalized(self):
        assert is_literal_match("hello    world", "hello world") is True
        assert is_literal_match("hello\nworld", "hello world") is True

    def test_is_literal_match_not_present(self):
        assert is_literal_match("goodbye", "hello world") is False
        assert is_literal_match("world hello", "hello world") is False

    def test_is_literal_match_empty_evidence(self):
        assert is_literal_match("", "hello world") is False
        assert is_literal_match("hello", "") is False


# ============================================================================
# P2: Lexical false friend detection tests
# ============================================================================


class TestLexical:
    """Testes do módulo lexical.py (P2 - False Friends fix)."""

    def test_detect_oracle_database(self):
        text = "We use Oracle Database for persistence."
        signals = detect_false_friend_signals(text, ["Oracle", "Proxy contract"])
        assert any(s.pattern == "Oracle" for s in signals)

    def test_detect_nginx_proxy(self):
        text = "The nginx reverse proxy forwards requests."
        signals = detect_false_friend_signals(text, ["Proxy contract"])
        assert any(s.pattern == "Proxy contract" for s in signals)

    def test_no_false_friends_when_not_candidate(self):
        text = "Oracle Database used here"
        signals = detect_false_friend_signals(text, ["Snapshotting"])
        assert not signals

    def test_no_false_friends_not_in_text(self):
        text = "We implemented the pattern properly"
        signals = detect_false_friend_signals(text, ["Oracle", "Proxy contract"])
        assert not signals


# ============================================================================
# P1: Confidence calculation tests
# ============================================================================


class TestConfidence:
    """Testes do módulo confidence.py (P1 - Excessive Confidence fix)."""

    def test_confidence_insufficient_context(self):
        payload = {"verdict": "insufficient_context"}
        assert compute_stage2_confidence(payload) == "low"

    def test_confidence_no_without_evidence(self):
        payload = {"verdict": "no", "evidence_text": ""}
        assert compute_stage2_confidence(payload) == "low"

    def test_confidence_no_with_evidence(self):
        payload = {"verdict": "no", "evidence_text": "x" * 20}
        assert compute_stage2_confidence(payload) == "medium"

    def test_confidence_uncertain_without_evidence(self):
        payload = {"verdict": "uncertain", "evidence_text": ""}
        assert compute_stage2_confidence(payload) == "low"

    def test_confidence_uncertain_with_evidence(self):
        payload = {"verdict": "uncertain", "evidence_text": "x" * 20}
        assert compute_stage2_confidence(payload) == "medium"

    def test_confidence_yes_without_evidence(self):
        payload = {"verdict": "yes", "evidence_text": ""}
        assert compute_stage2_confidence(payload) == "low"

    def test_confidence_yes_with_weak_matches(self):
        payload = {
            "verdict": "yes",
            "evidence_text": "x" * 20,
            "mechanism_match": False,
            "scope_match": True,
            "focus_match": True,
        }
        assert compute_stage2_confidence(payload) == "medium"

    def test_confidence_yes_with_strong_evidence(self):
        payload = {
            "verdict": "yes",
            "evidence_text": "x" * 60,
            "mechanism_match": True,
            "scope_match": True,
            "focus_match": True,
        }
        assert compute_stage2_confidence(payload) == "high"

    def test_confidence_yes_with_false_friend(self):
        payload = {
            "verdict": "yes",
            "evidence_text": "x" * 60,
            "false_friend_detected": "yes",
            "mechanism_match": True,
            "scope_match": True,
            "focus_match": True,
        }
        assert compute_stage2_confidence(payload) == "low"

    def test_confidence_yes_superficial_mention(self):
        payload = {
            "verdict": "yes",
            "evidence_text": "x" * 60,
            "adoption_status": "superficial_mention",
            "mechanism_match": True,
            "scope_match": True,
            "focus_match": True,
        }
        assert compute_stage2_confidence(payload) == "low"


# ============================================================================
# P10: Unicode normalization (ZWJ preservation) tests
# ============================================================================


class TestUnicodeNormalization:
    """Testes de preservação de ZWJ em normalization.py (P10)."""

    def test_preserve_zwj_in_emoji(self):
        # ZWJ é U+200D (ZERO WIDTH JOINER)
        text = "Man technologist: 👨\u200d💻"
        normalized = normalize_unicode(text)
        # O ZWJ deveria ser preservado
        assert "\u200d" in normalized

    def test_remove_soft_hyphen(self):
        # Soft hyphen é U+00AD
        text = "soft\u00adhyphen"
        normalized = normalize_unicode(text)
        # O soft hyphen deveria ser removido
        assert "\u00ad" not in normalized

    def test_preserve_zwnj(self):
        # ZERO WIDTH NON-JOINER U+200C
        text = "Indic\u200cscript"
        normalized = normalize_unicode(text)
        # ZWNJ deveria ser preservado
        assert "\u200c" in normalized


# ============================================================================
# P3+P4: Schema validation tests
# ============================================================================


class TestSchemaNormalization:
    """Testes de normalização de schema com P3 (mechanism_match) e P4 (UNCERTAIN)."""

    def test_normalize_stage2_adds_mechanism_match_defaults(self):
        payload = {
            "verdict": "yes",
            "evidence_text": "some evidence",
            "evidence_location": ["body"],
            "justification": "reason",
            "adoption_status": "implemented_existing",
            "false_friend_detected": "no",
            "pattern_challenge_categories": [],
            "confidence": "medium",
            "alternative_pattern": "",
            "overlap_with": [],
        }
        catalog = PatternCatalog(
            dataframe=None,
            names=["Pattern A"],
            by_name={"Pattern A": {"pattern": "Pattern A", "category": "C", "subcategory": "S", "description": "D"}},
            compact_catalog="",
            full_catalog="",
        )
        result = normalize_stage2(payload, catalog)
        assert "mechanism_match" in result
        assert "scope_match" in result
        assert "focus_match" in result

    def test_normalize_stage2_downgrades_yes_to_uncertain_when_no_focus_match(self):
        payload = {
            "verdict": "yes",
            "mechanism_match": True,
            "scope_match": True,
            "focus_match": False,
            "evidence_text": "some evidence",
            "evidence_location": ["body"],
            "justification": "reason",
            "adoption_status": "implemented_existing",
            "false_friend_detected": "no",
            "pattern_challenge_categories": [],
            "confidence": "high",
            "alternative_pattern": "",
            "overlap_with": [],
        }
        catalog = PatternCatalog(
            dataframe=None,
            names=["Pattern A"],
            by_name={"Pattern A": {"pattern": "Pattern A", "category": "C", "subcategory": "S", "description": "D"}},
            compact_catalog="",
            full_catalog="",
        )
        result = normalize_stage2(payload, catalog)
        assert result["verdict"] == "uncertain"
        assert "auto-downgraded" in result["justification"]


# ============================================================================
# Feature 1: Checkpointing tests
# ============================================================================


class TestCheckpoint:
    """Testes do módulo checkpoint.py (Feature 1 - Stateful Checkpointing)."""

    def test_checkpoint_new(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.json"
            cp = Checkpoint(cp_path, stage="stage1", run_id="test_run")
            assert cp.completed_count() == 0
            assert cp.failed_count() == 0

    def test_checkpoint_mark_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.json"
            cp = Checkpoint(cp_path, stage="stage1", run_id="test_run")
            cp.mark_completed("id1")
            assert cp.is_completed("id1")
            assert cp.completed_count() == 1

    def test_checkpoint_persists_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.json"
            cp1 = Checkpoint(cp_path, stage="stage1", run_id="test_run")
            cp1.mark_completed("id1")
            cp1.mark_failed("id2")
            
            # Reload
            cp2 = Checkpoint(cp_path, stage="stage1", run_id="test_run")
            assert cp2.is_completed("id1")
            assert "id2" in cp2._state.failed_ids
            assert cp2.completed_count() == 1
            assert cp2.failed_count() == 1

    def test_checkpoint_skip_already_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.json"
            cp = Checkpoint(cp_path, stage="stage1", run_id="test_run")
            cp.mark_completed("item_1")
            assert cp.is_completed("item_1")
            assert not cp.is_completed("item_2")


# ============================================================================
# Integration mock tests (without real API calls)
# ============================================================================


class TestMockStage2Integration:
    """Testes de integração mockeados para Stage 2."""

    def test_stage2_confidence_override_on_mock_response(self):
        """Verifica que a confiança reportada é sobrescrita com valor computado."""
        mock_payload = {
            "verdict": "yes",
            "mechanism_match": True,
            "scope_match": True,
            "focus_match": True,
            "evidence_text": "good evidence text here",
            "evidence_location": ["body"],
            "justification": "reason",
            "adoption_status": "implemented_existing",
            "false_friend_detected": "no",
            "pattern_challenge_categories": [],
            "confidence": "low",  # Mock low confidence
            "alternative_pattern": "",
            "overlap_with": [],
        }
        catalog = PatternCatalog(
            dataframe=None,
            names=["Pattern A"],
            by_name={"Pattern A": {"pattern": "Pattern A", "category": "C", "subcategory": "S", "description": "D"}},
            compact_catalog="",
            full_catalog="",
        )
        normalized = normalize_stage2(mock_payload, catalog)
        
        # Compute the confidence that stages.py would apply
        computed = compute_stage2_confidence(normalized)
        # Should be 'medium' or 'high' despite mock reporting 'low'
        assert computed in {"medium", "high"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
