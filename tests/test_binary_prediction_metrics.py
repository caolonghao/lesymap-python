import numpy as np

from lesymap.utils.metrics import evaluate_binary_predictions


def test_binary_prediction_metrics_selects_threshold_from_scores():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])

    metrics = evaluate_binary_predictions(y_true, scores, threshold="youden")

    assert metrics["threshold"] == 0.6
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == {
        "tn": 3,
        "fp": 0,
        "fn": 0,
        "tp": 3,
    }


def test_binary_prediction_metrics_supports_fixed_threshold():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([-0.2, 0.4, 0.3, 1.2])

    metrics = evaluate_binary_predictions(y_true, scores, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert metrics["confusion_matrix"] == {
        "tn": 2,
        "fp": 0,
        "fn": 1,
        "tp": 1,
    }
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0


def test_binary_prediction_metrics_rejects_nonbinary_labels():
    y_true = np.array([0, 1, 2])
    scores = np.array([0.1, 0.5, 0.9])

    try:
        evaluate_binary_predictions(y_true, scores)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Expected non-binary labels to be rejected")
