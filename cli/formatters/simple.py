"""Simple text-based output formatter."""

from .base import OutputFormatter


class SimpleFormatter(OutputFormatter):
    """Simple text-based formatter with dots for quick feedback."""

    def __init__(self, verbose: bool = False, use_color: bool = True):
        """Initialize simple formatter."""
        super().__init__(verbose=verbose, use_color=use_color)
        self._current_line_length = 0
        self._max_line_length = 70

    def test_started(self, test_name: str, index: int, total: int) -> None:
        """Print test start (verbose mode only)."""
        if self.verbose:
            print(f"\n[{index + 1}/{total}] {test_name}")

    def test_passed(self, test_name: str) -> None:
        """Print pass indicator."""
        if self.verbose:
            print(self.green("  ✓ PASSED"))
        else:
            self._print_dot(self.green("."))

    def test_failed(self, test_name: str, error_message: str) -> None:
        """Print failure indicator and error."""
        if self.verbose:
            print(self.red(f"  ✗ FAILED: {error_message}"))
        else:
            self._print_dot(self.red("F"))

    def test_skipped(self, test_name: str, reason: str) -> None:
        """Print skip indicator."""
        if self.verbose:
            print(self.yellow(f"  - SKIPPED: {reason}"))
        else:
            self._print_dot(self.yellow("s"))

    def summary(self, passed: int, failed: int, skipped: int, total: int) -> None:
        """Print test summary."""
        # Print newline after dots if in non-verbose mode
        if not self.verbose and self._current_line_length > 0:
            print()

        print("\n" + "=" * 70)

        # Print overall status
        if failed > 0:
            status = self.red(self.bold("FAILED"))
        elif total == 0:
            status = self.yellow("NO TESTS RAN")
        else:
            status = self.green(self.bold("PASSED"))

        print(f"{status}")
        print("=" * 70)

        # Print counts
        print(f"Tests run: {total}")
        if passed > 0:
            print(self.green(f"Passed: {passed}"))
        if failed > 0:
            print(self.red(f"Failed: {failed}"))
        if skipped > 0:
            print(self.yellow(f"Skipped: {skipped}"))

        print()

    def _print_dot(self, dot: str) -> None:
        """
        Print a single character indicator (dot, F, s, etc.).

        Wraps to new line after max_line_length characters.
        """
        print(dot, end='', flush=True)
        self._current_line_length += 1

        if self._current_line_length >= self._max_line_length:
            print()  # New line
            self._current_line_length = 0
