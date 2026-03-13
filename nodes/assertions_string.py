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

    SKILL_DOC = """
Verifies that a string contains an expected substring. Fails if the substring is not
found. This is an output node.

### Inputs

- `text` (ANY) — The string to search in. Must be a string type.
- `substring` (STRING) — The substring to search for. Must not be empty. Supports
  multiline. Default: `""`.
- `case_sensitive` (BOOLEAN) — Whether matching is case-sensitive. Default: `false`.

### When to Use

- Verify prompt builders inject expected tags (e.g. output contains `"masterpiece"`).
- Check string concatenation includes both input strings.
- Validate template substitution replaced placeholders with expected values.
- Confirm metadata extractors return expected keys or values.

### Tips

- Case-insensitive by default — set `case_sensitive=true` when exact casing matters.
- Error message shows a preview of the text (truncated to 200 chars) for debugging.
- For checking that a substring is NOT present, use `AssertStringNotContains` instead.
- For exact full-string matching, use `AssertStringMatch` with `mode="exact"`.

### Example

Verify a quality tag injector adds expected text:
```
[PromptBuilder(input="a cat")] → [AssertStringContains(substring="masterpiece")]
```
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


class AssertStringNotContains(io.ComfyNode):
    """Output node that checks a string does NOT contain a forbidden pattern.

    Modes:
      literal — Simple substring search. Fails if the substring appears anywhere
                in the text. Use for single known-bad strings.
      regex   — Python regex search. Fails if the pattern matches anywhere in the
                text (uses re.search). Use for matching multiple forbidden patterns
                at once, e.g. r"(I cannot|I'm sorry|error|exception)".
      llm_guard — Built-in check for common LLM failure patterns. The pattern field
                  is ignored. Checks for: empty/whitespace-only output, refusal
                  phrases, apology hedging, error/exception markers, and
                  safety disclaimers. Case-insensitive.

    Example test patterns:
      - LLM response → llm_guard mode (catches refusals, errors, empty output)
      - Prompt sanitizer → literal mode, check for banned word
      - Template resolver → regex mode, r"\\{\\w+\\}" to catch unresolved placeholders
      - Multi-pattern ban → regex mode, r"(badword1|badword2|badword3)"
    """

    SKILL_DOC = """
Verifies that a string does NOT contain a forbidden pattern. Supports three matching
modes. This is an output node.

### Inputs

- `text` (ANY) — The string to check. Must be a string type.
- `pattern` (STRING) — The forbidden pattern. Ignored in `llm_guard` mode. Supports
  multiline. Default: `""`.
- `mode` (COMBO) — Matching strategy:
  - `"literal"` — Simple substring search. Fails if `pattern` appears anywhere.
  - `"regex"` — Python regex via `re.search()`. Fails if pattern matches anywhere.
  - `"llm_guard"` — Built-in patterns (ignores `pattern` field). Catches:
    - Empty/whitespace-only output
    - Refusal phrases: "I cannot", "I can't", "I'm sorry", "I apologize",
      "I'm unable to", "I am unable to", "I'm not able to"
    - AI disclaimers: "as an AI", "as a language model"
    - Error markers: "error:", "exception:", "traceback", "syntax error", "runtime error"
  Default: `"literal"`.
- `case_sensitive` (BOOLEAN) — Controls case sensitivity for `literal` and `regex` modes.
  `llm_guard` is always case-insensitive. Default: `false`.

### When to Use

- **`llm_guard` mode**: As a baseline assertion for ANY LLM/AI text generation node.
  Catches the most common failure modes in a single assertion.
- **`literal` mode**: Check for a specific banned word or phrase.
- **`regex` mode**: Check for unresolved template placeholders (`r"\\{\\w+\\}"`),
  multiple banned patterns (`r"(word1|word2|word3)"`), or structural issues.

### Tips

- For LLM nodes, layer `llm_guard` with `AssertStringLength` for comprehensive coverage.
- Error messages include ±50 chars of context around the match for debugging.
- In `llm_guard` mode, the `pattern` field is completely ignored.

### Example

Baseline assertion for an LLM captioning node:
```
[ImageCaptioner(image=...)] → [AssertStringNotContains(mode="llm_guard")]
```

Check that a template resolver removed all placeholders:
```
[TemplateResolver(template="{subject} in {style}")] → [AssertStringNotContains(pattern="{", mode="literal")]
```
"""

    LLM_GUARD_PATTERNS = [
        r"^\s*$",
        r"\bI cannot\b",
        r"\bI can't\b",
        r"\bI'm sorry\b",
        r"\bI apologize\b",
        r"\bI'm unable to\b",
        r"\bI am unable to\b",
        r"\bI'm not able to\b",
        r"\bas an AI\b",
        r"\bas a language model\b",
        r"\berror\s*:",
        r"\bexception\s*:",
        r"\btraceback\b",
        r"\bsyntax\s*error\b",
        r"\bruntime\s*error\b",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertStringNotContains",
            display_name="Assert String Not Contains",
            category="testing",
            description="Check that a string does NOT contain a forbidden pattern (literal, regex, or built-in LLM guard)",
            inputs=[
                io.AnyType.Input("text"),
                io.String.Input(
                    "pattern",
                    default="",
                    multiline=True,
                ),
                io.Combo.Input(
                    "mode",
                    options=["literal", "regex", "llm_guard"],
                    default="literal",
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
        """Check that text does NOT contain the forbidden pattern."""
        import re

        if not isinstance(text, str):
            raise ValueError(f"Input is not a string, got {type(text).__name__}")

        if mode == "llm_guard":
            combined = "|".join(f"({p})" for p in cls.LLM_GUARD_PATTERNS)
            match = re.search(combined, text, re.IGNORECASE)
            if match:
                pos = match.start()
                start = max(0, pos - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                matched_text = match.group() if match.group().strip() else "(empty/whitespace)"
                raise ValueError(
                    f"LLM guard triggered:\n"
                    f"Matched: {repr(matched_text)}\n"
                    f"At position: {pos}\n"
                    f"Context: {repr(context)}"
                )

            PromptServer.instance.send_progress_text(
                f"✅ Test Passed\nLLM guard: no refusals/errors found in text ({len(text)} chars)",
                cls.hidden.unique_id
            )
            return io.NodeOutput()

        if not pattern:
            raise ValueError("Pattern to check against is empty")

        if mode == "literal":
            haystack = text if case_sensitive else text.lower()
            needle = pattern if case_sensitive else pattern.lower()

            if needle in haystack:
                pos = haystack.index(needle)
                start = max(0, pos - 50)
                end = min(len(text), pos + len(pattern) + 50)
                context = text[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                raise ValueError(
                    f"String contains forbidden substring:\n"
                    f"Found: {repr(pattern)}\n"
                    f"At position: {pos}\n"
                    f"Case sensitive: {case_sensitive}\n"
                    f"Context: {repr(context)}"
                )

        elif mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            match = re.search(pattern, text, flags)
            if match:
                pos = match.start()
                start = max(0, pos - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                raise ValueError(
                    f"String matches forbidden regex pattern:\n"
                    f"Pattern: {repr(pattern)}\n"
                    f"Matched: {repr(match.group())}\n"
                    f"At position: {pos}\n"
                    f"Case sensitive: {case_sensitive}\n"
                    f"Context: {repr(context)}"
                )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nMode: {mode}, pattern not found in text ({len(text)} chars)",
            cls.hidden.unique_id
        )
        return io.NodeOutput()


class AssertStringLength(io.ComfyNode):
    """Output node that checks a string's length is within an expected range.

    Useful for non-deterministic outputs where exact content varies but length
    should fall within reasonable bounds. Catches empty, truncated, or runaway outputs.

    Example test patterns:
      - LLM caption → expect 50-500 chars (catches empty or degenerate responses)
      - Prompt template resolution → expect roughly same length as template
      - Text summarizer → expect output shorter than input (use with known input length)
      - Filename generator → expect 10-100 chars
    """

    SKILL_DOC = """
Checks that a string's length (in characters) falls within `[min_length, max_length]`.
Catches empty, truncated, or runaway outputs. This is an output node.

### Inputs

- `text` (ANY) — The string to measure. Must be a string type.
- `min_length` (INT) — Minimum acceptable length. Range: 0–10000000. Default: `0`.
- `max_length` (INT) — Maximum acceptable length. Range: 0–10000000. Default: `10000`.

### When to Use

- For non-deterministic outputs (LLM, randomized text) where exact content varies
  but length should be reasonable.
- Catch empty outputs (`min_length=1`).
- Catch degenerate/runaway generation (`max_length=5000`).
- Validate text summarizers produce shorter output than input.
- Use alongside `AssertStringNotContains(mode="llm_guard")` for comprehensive LLM testing.

### Tips

- Set `min_length=1` to catch empty string failures.
- The error message includes a preview of the text (truncated to 100 chars).
- For exact length matching, set `min_length` = `max_length`.

### Example

Verify an LLM caption produces substantive output:
```
[ImageCaptioner(image=...)] → [AssertStringLength(min_length=20, max_length=500)]
```
"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AssertStringLength",
            display_name="Assert String Length",
            category="testing",
            description="Check that a string's length is within min/max bounds",
            inputs=[
                io.AnyType.Input("text"),
                io.Int.Input(
                    "min_length",
                    default=0,
                    min=0,
                    max=10000000,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "max_length",
                    default=10000,
                    min=0,
                    max=10000000,
                    step=1,
                    display_mode=io.NumberDisplay.number,
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
        min_length: int,
        max_length: int,
    ) -> io.NodeOutput:
        """Check that text length is within bounds."""

        if not isinstance(text, str):
            raise ValueError(f"Input is not a string, got {type(text).__name__}")

        length = len(text)
        errors = []

        if length < min_length:
            errors.append(f"Length {length} is below minimum {min_length}")
        if length > max_length:
            errors.append(f"Length {length} exceeds maximum {max_length}")

        if errors:
            preview = text[:100] + "..." if len(text) > 100 else text
            raise ValueError(
                "String length assertion failed:\n" + "\n".join(errors) +
                f"\nActual length: {length} chars\n"
                f"Expected: [{min_length}, {max_length}]\n"
                f"Preview: {repr(preview)}"
            )

        PromptServer.instance.send_progress_text(
            f"✅ Test Passed\nLength: {length} chars (expected [{min_length}, {max_length}])",
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

    SKILL_DOC = """
Checks that a string matches an expected pattern exactly or via regex. This is an
output node.

### Inputs

- `text` (ANY) — The string to validate. Must be a string type.
- `pattern` (STRING) — The expected value or regex pattern. Supports multiline.
  Default: `""`.
- `mode` (COMBO) — Matching strategy:
  - `"exact"` — Full string equality (entire string must match).
  - `"regex"` — Python `re.fullmatch()` (entire string must match the regex pattern).
  Default: `"exact"`.
- `case_sensitive` (BOOLEAN) — Controls case sensitivity. Default: `false`.

### When to Use

- **`exact` mode**: For deterministic string operations where the entire output is known
  (string replace, concatenation, formatting, fixed templates).
- **`regex` mode**: For validating output structure when exact content varies
  (wildcard resolvers, timestamp generators, filename builders).

### Tips

- Uses `re.fullmatch()` in regex mode — the ENTIRE string must match the pattern,
  not just a substring. Use `AssertStringContains` for substring checks.
- Case-insensitive by default in both modes.
- Error messages show text preview (truncated to 200 chars) for debugging.
- For just checking a substring is present, prefer `AssertStringContains`.

### Example

Verify a string replace operation:
```
[StringReplace(text="a photo of a dog", find="dog", replace="cat")] → [AssertStringMatch(pattern="a photo of a cat", mode="exact")]
```

Verify a date formatter produces correct format:
```
[DateFormatter] → [AssertStringMatch(pattern="\\d{4}-\\d{2}-\\d{2}", mode="regex")]
```
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
