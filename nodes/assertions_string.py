from typing import Any

from comfy_api.v0_0_2 import io
from server import PromptServer


class AssertStringContains(io.ComfyNode):
    """Output node that checks if a string contains an expected substring.

    Example test patterns:
      - Prompt builder added quality tags → assert output contains "masterpiece"
      - String concatenation → assert output contains both input strings
      - Template substitution → assert output contains the substituted value
      - Metadata extractor → assert output contains expected EXIF key
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertStringContains",
            display_name="Assert String Contains",
            category="testing",
            description="Check that a string contains an expected substring",
            inputs=[
                io.AnyType.Input("text"),
                io.String.Input(
                    "substring",
                    default="",
                    multiline=True,
                ),
                io.Boolean.Input(
                    "case_sensitive",
                    default=False,
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        text: Any,
        substring: str,
        case_sensitive: bool,
    ) -> io.NodeOutput:
        """Check that text contains substring."""

        if not isinstance(text, str):
            raise ValueError(f"Input is not a string, got {type(text).__name__}")

        if not substring:
            raise ValueError("Substring to search for is empty")

        haystack = text if case_sensitive else text.lower()
        needle = substring if case_sensitive else substring.lower()

        if needle not in haystack:
            # Show a truncated preview if the text is very long
            preview = text[:200] + "..." if len(text) > 200 else text
            raise ValueError(
                f"String does not contain expected substring:\n"
                f"Looking for: {repr(substring)}\n"
                f"Case sensitive: {case_sensitive}\n"
                f"In text ({len(text)} chars): {repr(preview)}"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nFound {repr(substring)} in text ({len(text)} chars)",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertStringMatch(io.ComfyNode):
    """Output node that checks if a string matches a pattern exactly or via regex.

    Modes:
      exact — Full string equality. Use for deterministic string operations where the
              entire output is known (e.g. string replace, concatenation, formatting).
      regex — Python regex match against the full string (uses re.fullmatch).
              Use for validating structure when exact content varies
              (e.g. wildcard resolvers, timestamp generators).

    Example test patterns:
      - String replace "dog" → "cat" → exact match "a photo of a cat"
      - Wildcard resolver → regex r"\\w+ sky" to verify structure
      - Filename builder → regex r".+_\\d{4}\\.png" to verify format
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertStringMatch",
            display_name="Assert String Match",
            category="testing",
            description="Check that a string matches a pattern (exact or regex)",
            inputs=[
                io.AnyType.Input("text"),
                io.String.Input(
                    "pattern",
                    default="",
                    multiline=True,
                ),
                io.Combo.Input(
                    "mode",
                    options=["exact", "regex"],
                    default="exact",
                ),
                io.Boolean.Input(
                    "case_sensitive",
                    default=False,
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        text: Any,
        pattern: str,
        mode: str,
        case_sensitive: bool,
    ) -> io.NodeOutput:
        """Check that text matches the expected pattern."""
        import re

        if not isinstance(text, str):
            raise ValueError(f"Input is not a string, got {type(text).__name__}")

        if not pattern and mode == "regex":
            raise ValueError("Regex pattern is empty")

        if mode == "exact":
            a = text if case_sensitive else text.lower()
            b = pattern if case_sensitive else pattern.lower()
            if a != b:
                preview_text = text[:200] + "..." if len(text) > 200 else text
                preview_pat = pattern[:200] + "..." if len(pattern) > 200 else pattern
                raise ValueError(
                    f"String does not match expected value:\n"
                    f"Expected: {repr(preview_pat)}\n"
                    f"Got:      {repr(preview_text)}\n"
                    f"Case sensitive: {case_sensitive}"
                )
        else:  # regex
            flags = 0 if case_sensitive else re.IGNORECASE
            if not re.fullmatch(pattern, text, flags):
                preview = text[:200] + "..." if len(text) > 200 else text
                raise ValueError(
                    f"String does not match regex pattern:\n"
                    f"Pattern: {repr(pattern)}\n"
                    f"Text ({len(text)} chars): {repr(preview)}\n"
                    f"Case sensitive: {case_sensitive}"
                )

        preview = text[:100] + "..." if len(text) > 100 else text
        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nMode: {mode}, matched {repr(preview)} ({len(text)} chars)",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertStringHash(io.ComfyNode):
    """Output node that compares a string's SHA-256 hash against an expected value.

    Designed for large text outputs (LLM responses, metadata dumps, full captions)
    where storing and comparing the full text in the workflow JSON is impractical.

    Usage:
      1. First run: leave expected_hash empty → the node shows the calculated hash.
      2. Copy the hash into expected_hash → subsequent runs validate against it.

    This mirrors the workflow of AssertImageMatch where you accept the hash after
    verifying the output is correct.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertStringHash",
            display_name="Assert String Hash",
            category="testing",
            description="Compare a string's SHA-256 hash against an expected value (for large text outputs)",
            inputs=[
                io.AnyType.Input("text"),
                io.String.Input(
                    "expected_hash",
                    default="",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        text: Any,
        expected_hash: str,
    ) -> io.NodeOutput:
        """Compare string hash against expected value."""
        import hashlib

        if not isinstance(text, str):
            raise ValueError(f"Input is not a string, got {type(text).__name__}")

        calculated = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if not expected_hash or expected_hash.strip() == "":
            PromptServer.instance.send_progress_text(
                f"Hash (no expected value set):\n{calculated}\n"
                f"Text length: {len(text)} chars\n"
                f"Copy this hash into expected_hash to enable assertion.",
                cls.hidden.unique_id
            )
            return io.NodeOutput()

        expected_clean = expected_hash.strip().lower()
        if calculated != expected_clean:
            preview = text[:100] + "..." if len(text) > 100 else text
            raise ValueError(
                f"String hash mismatch:\n"
                f"Expected: {expected_clean}\n"
                f"Got:      {calculated}\n"
                f"Text length: {len(text)} chars\n"
                f"Preview: {repr(preview)}"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nHash: {calculated}\nText length: {len(text)} chars",
            cls.hidden.unique_id
        )
        return io.NodeOutput()
