from icl_sentiment.evaluation.metrics import evaluate_predictions, normalize_label


def test_normalize_label_handles_case_and_whitespace():
    assert normalize_label("  Positive  ", ["positive", "neutral", "negative"]) == "positive"


def test_normalize_label_extracts_from_verbose_response():
    assert normalize_label("Sentiment: negative.", ["positive", "neutral", "negative"]) == "negative"


def test_normalize_label_unparseable_is_invalid():
    assert normalize_label("I cannot decide", ["positive", "neutral", "negative"]) == "invalid"


def test_normalize_label_none_is_invalid():
    assert normalize_label(None, ["positive", "neutral", "negative"]) == "invalid"


def test_evaluate_predictions_perfect_score():
    gold = ["positive", "neutral", "negative"]
    pred = ["positive", "neutral", "negative"]
    result = evaluate_predictions(gold, pred)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.n_invalid == 0


def test_evaluate_predictions_partial_score():
    gold = ["positive", "neutral", "negative", "positive"]
    pred = ["positive", "positive", "negative", "neutral"]
    result = evaluate_predictions(gold, pred)
    assert result.accuracy == 0.5
    assert 0 < result.macro_f1 < 1


def test_evaluate_predictions_counts_invalid_responses():
    gold = ["positive", "neutral"]
    pred = ["positive", "I don't know"]
    result = evaluate_predictions(gold, pred)
    assert result.n_invalid == 1
    assert result.accuracy == 0.5


def test_evaluate_predictions_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        evaluate_predictions(["positive"], ["positive", "negative"])


def test_per_class_frame_has_expected_index():
    result = evaluate_predictions(["positive", "negative"], ["positive", "negative"])
    assert list(result.per_class.index) == ["positive", "neutral", "negative"]
