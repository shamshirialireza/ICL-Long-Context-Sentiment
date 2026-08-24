from icl_sentiment.prompting.examples import FewShotExample
from icl_sentiment.prompting.templates import PromptBuilder


def test_zero_shot_prompt_has_no_examples_section():
    builder = PromptBuilder(labels=("positive", "neutral", "negative"), document_noun="comment")
    prompt = builder.build("Great service!")
    assert "Great service!" in prompt
    assert "Sentiment:" not in prompt
    assert "positive or neutral or negative" in prompt


def test_few_shot_prompt_includes_examples_and_target():
    builder = PromptBuilder(labels=("positive", "negative"), document_noun="review")
    examples = [
        FewShotExample(text="Loved it", label="positive"),
        FewShotExample(text="Hated it", label="negative"),
    ]
    prompt = builder.build("It was fine", examples=examples)
    assert 'review: "Loved it"' in prompt
    assert "Sentiment: positive" in prompt
    assert "It was fine" in prompt


def test_document_noun_is_configurable():
    builder = PromptBuilder(document_noun="post")
    prompt = builder.build("hello world")
    assert "post" in prompt.lower()
