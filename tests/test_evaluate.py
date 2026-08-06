import evaluate_pipeline as ep


def test_split_patterns_accepts_json_and_pipe():
    assert ep.split_patterns('["Oracle", "Proxy contract"]') == ["Oracle", "Proxy contract"]
    assert ep.split_patterns("Oracle|Proxy contract") == ["Oracle", "Proxy contract"]


def test_binary_metrics():
    metrics = ep.binary_metrics(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"])
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
