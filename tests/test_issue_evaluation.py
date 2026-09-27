import pandas as pd
import pytest

from compare_pipeline_runs import _classification, _write_csv
from evaluate_pipeline import binary_metrics
from llm_pipeline.aggregation import aggregate_issue_results
from llm_pipeline.issue_evaluation import human_issue_reference, issue_level_binary


def human(labels):
    return pd.DataFrame(
        {
            "repository": ["org/repo"] * len(labels),
            "issue_number": [str(i + 1) for i in range(len(labels))],
            "relevant_to_pattern_study": labels,
        }
    )


def predictions(values):
    return pd.DataFrame(
        {
            "repository": ["org/repo"] * len(values),
            "issue_number": [str(i + 1) for i in range(len(values))],
            "relevant_to_pattern_study": ["yes" if value else "no" for value in values],
        }
    )


def test_issue_confusion_and_exclusions():
    metrics, rows = issue_level_binary(
        human(["yes", "yes", "no", "no", "uncertain", "insufficient_context"]),
        predictions([True, False, True, False, True, True]),
    )
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["n"] == 4
    assert metrics["excluded_uncertain"] == 1
    assert metrics["excluded_insufficient_context"] == 1
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert rows.classification.tolist()[-2:] == [
        "EXCLUDED_UNCERTAIN",
        "EXCLUDED_INSUFFICIENT_CONTEXT",
    ]


def test_issue_metrics_zero_predicted_positives():
    metrics, _ = issue_level_binary(human(["yes", "no"]), predictions([False, False]))
    assert metrics["tp"] == 0 and metrics["fn"] == 1 and metrics["tn"] == 1
    assert metrics["precision"] is None
    assert metrics["recall"] == 0
    assert metrics["negative_predictive_value"] == 0.5


def test_issue_metrics_zero_human_positives():
    metrics, _ = issue_level_binary(human(["no", "no"]), predictions([True, False]))
    assert metrics["recall"] is None
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] is None


def test_duplicate_issue_keys_fail_explicitly():
    frame = human(["yes", "no"])
    frame["issue_number"] = "1"
    with pytest.raises(ValueError, match="duplicadas"):
        human_issue_reference(frame)


@pytest.mark.parametrize("label", ["", "maybe"])
def test_missing_or_unknown_human_label_fails(label):
    with pytest.raises(ValueError, match="labels ausentes/desconhecidos"):
        human_issue_reference(human([label]))


def test_aggregation_uses_any_stage2_yes_not_stage1_candidate():
    stage1 = pd.DataFrame(
        [
            {"repository": "org/repo", "issue_number": "1", "request_status": "succeeded", "candidate_count": 2, "context_status": "sufficient"},
            {"repository": "org/repo", "issue_number": "2", "request_status": "succeeded", "candidate_count": 1, "context_status": "sufficient"},
        ]
    )
    stage2 = pd.DataFrame(
        [
            {"repository": "org/repo", "issue_number": "1", "pattern": "A", "request_status": "succeeded", "verdict": "no"},
            {"repository": "org/repo", "issue_number": "1", "pattern": "B", "request_status": "succeeded", "verdict": "yes"},
            {"repository": "org/repo", "issue_number": "2", "pattern": "A", "request_status": "succeeded", "verdict": "no"},
        ]
    )
    result = aggregate_issue_results(stage1, stage2)
    assert result.relevant_to_pattern_study.tolist() == ["yes", "no"]
    assert result.patterns_confirmed.tolist() == ["B", ""]


def test_binary_metrics_preserves_old_keys_and_adds_rates():
    metrics = binary_metrics(["yes", "no"], ["yes", "no"])
    for key in ["n", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"]:
        assert key in metrics
    assert metrics["specificity"] == 1
    assert metrics["false_positive_rate"] == 0
    assert metrics["false_negative_rate"] == 0
    assert metrics["balanced_accuracy"] == 1


def test_fp_fn_classification_and_divergence_csv(tmp_path):
    assert _classification("no", True) == "FP"
    assert _classification("yes", False) == "FN"
    path = tmp_path / "divergences.csv"
    _write_csv(path, [{"repository": "org/repo", "issue_number": "1", "classification": "FP"}])
    loaded = pd.read_csv(path)
    assert loaded.loc[0, "classification"] == "FP"
