#Prompt construction for zero-shot and few-shot sentiment classification

"""Domain-neutral: the document noun ("comment", "review", "post", ...) and the
label set are configurable instead of being baked into the prompt text."""

from __future__ import annotations

from collections.abc import Sequence

from icl_sentiment.prompting.examples import FewShotExample

DEFAULT_LABELS = ("positive", "neutral", "negative")

class PromptBuilder:
    """Builds classification prompts from a document, labels, and optional examples."""

    def __init__(
        self,
        labels: Sequence[str] = DEFAULT_LABELS,
        document_noun: str = "text",
        system_instruction: str | None = None,
    ):
        self.labels = tuple(labels)
        self.document_noun = document_noun
        self.system_instruction = (
            system_instruction
            or "You are a precise sentiment analyzer. Respond with exactly one word."
        )

    @property
    def label_options(self) -> str:
        return " or ".join(self.labels)

    def build(self, text: str, examples: Sequence[FewShotExample] = ()) -> str:
        noun = self.document_noun
        header = (
            f"What is the overall sentiment expressed in the following {noun}?\n"
            f"Select only one sentiment value from {self.label_options}. "
            f"Return only the sentiment value in lowercase letters."
        )

        if not examples:
            return f"{header}\n\n{noun}: {text}"

        demonstrations = "\n\n".join(
            f'{noun}: "{example.text}"\nSentiment: {example.label}' for example in examples
        )
        return (
            f"{header}\n"
            f"Here are labeled examples to guide your answer:\n\n"
            f"{demonstrations}\n\n"
            f"Given these examples, what is the sentiment of the following {noun}?\n"
            f"{noun}: {text}"
        )
