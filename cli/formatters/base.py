"""Abstract base class for output formatters."""

from abc import ABC, abstractmethod
from typing import List

import sys
import os


class OutputFormatter(ABC):
    """Abstract base class for test result formatters."""

    def __init__(self, verbose: bool = False, use_color: bool = True):
        """
        Initialize formatter.

        Args:
            verbose: Enable verbose output
            use_color: Use ANSI color codes (auto-detected if terminal)
        """
        self.verbose = verbose
        # Auto-detect color support
        self.use_color = use_color and self._supports_color()

    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        # Check if stdout is a terminal
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False

        # Check for TERM environment variable
        term = os.environ.get('TERM', '')
        if term == 'dumb':
            return False

        # Assume color support for most terminals
        return True

    @abstractmethod
    def test_started(self, test_name: str, index: int, total: int) -> None:
        """
        Called when a test starts execution.

        Args:
            test_name: Name of the test
            index: Index of current test (0-based)
            total: Total number of tests
        """
        pass

    @abstractmethod
    def test_passed(self, test_name: str) -> None:
        """
        Called when a test passes.

        Args:
            test_name: Name of the test
        """
        pass

    @abstractmethod
    def test_failed(self, test_name: str, error_message: str) -> None:
        """
        Called when a test fails.

        Args:
            test_name: Name of the test
            error_message: Error or failure message
        """
        pass

    @abstractmethod
    def test_skipped(self, test_name: str, reason: str) -> None:
        """
        Called when a test is skipped.

        Args:
            test_name: Name of the test
            reason: Reason for skipping
        """
        pass

    @abstractmethod
    def summary(self, passed: int, failed: int, skipped: int, total: int) -> None:
        """
        Print test summary.

        Args:
            passed: Number of passed tests
            failed: Number of failed tests
            skipped: Number of skipped tests
            total: Total number of tests
        """
        pass

    def _colorize(self, text: str, color_code: str) -> str:
        """
        Apply ANSI color code to text if colors are enabled.

        Args:
            text: Text to colorize
            color_code: ANSI color code (e.g., '32' for green)

        Returns:
            Colorized text or plain text if colors disabled
        """
        if not self.use_color:
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def green(self, text: str) -> str:
        """Colorize text green."""
        return self._colorize(text, '32')

    def red(self, text: str) -> str:
        """Colorize text red."""
        return self._colorize(text, '31')

    def yellow(self, text: str) -> str:
        """Colorize text yellow."""
        return self._colorize(text, '33')

    def cyan(self, text: str) -> str:
        """Colorize text cyan."""
        return self._colorize(text, '36')

    def bold(self, text: str) -> str:
        """Make text bold."""
        return self._colorize(text, '1')
