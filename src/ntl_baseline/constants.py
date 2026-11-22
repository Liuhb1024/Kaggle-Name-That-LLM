"""Exact prediction labels from the competition's Evaluation page."""

LABELS: tuple[str, ...] = tuple(
    sorted(
        [
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "claude-fable-5",
            "claude-opus-4-6",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "qwen3.8-max",
            "glm-5.3",
            "glm-5.3-flash",
            "kimi-k3",
        ]
    )
)

DATA_FILES = ("train.csv", "test.csv", "sample_submission.parquet")


class ContractError(ValueError):
    """Invalid data, configuration, artifact or submission contract."""
