#!/usr/bin/env -S uv run
"""ComfyCI - CLI tool for running ComfyUI test workflows."""

import sys
from pathlib import Path
from typing import List

import click

from comfy_client import ComfyClient
from workflow_parser import discover_tests, WorkflowTest
from test_runner import TestRunner, SequentialExecutor, TestStatus
from formatters import SimpleFormatter


@click.command()
@click.argument('patterns', nargs=-1, required=True)
@click.option(
    '--target',
    default='localhost:8188',
    help='ComfyUI server address (host:port)',
    show_default=True,
)
@click.option(
    '--cpu',
    is_flag=True,
    help='Skip tests that require GPU',
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Enable verbose output',
)
@click.option(
    '-x', '--failfast',
    is_flag=True,
    help='Stop on first test failure',
)
@click.option(
    '--timeout',
    type=int,
    help='Override default timeout in seconds (default: 120 + test extraTime)',
)
@click.option(
    '--no-color',
    is_flag=True,
    help='Disable colored output',
)
def main(
    patterns: tuple,
    target: str,
    cpu: bool,
    verbose: bool,
    failfast: bool,
    timeout: int,
    no_color: bool,
):
    """
    Run ComfyUI test workflows.

    PATTERNS: Glob patterns for test workflow JSON files (e.g., ./tests/**/*.json)

    Examples:

        comfyci ./tests/**/*.json --target localhost:8188

        comfyci test1.json test2.json --cpu --verbose

        comfyci ./tests/*.json --failfast --timeout 300
    """
    # Initialize formatter
    formatter = SimpleFormatter(verbose=verbose, use_color=not no_color)

    try:
        # Discover test files
        if verbose:
            print(f"Discovering tests matching: {', '.join(patterns)}")

        tests = discover_tests(list(patterns))

        if not tests:
            print("No test files found matching the given patterns.")
            sys.exit(1)

        if verbose:
            print(f"Found {len(tests)} test(s)\n")

        # Filter tests based on GPU requirements
        if cpu:
            original_count = len(tests)
            tests = [t for t in tests if not t.requires_gpu]
            skipped_count = original_count - len(tests)

            if verbose and skipped_count > 0:
                print(f"Skipping {skipped_count} GPU-required test(s) (--cpu flag)\n")

            if not tests:
                print("All tests require GPU. Use without --cpu to run them.")
                sys.exit(0)

        # Apply custom timeout if specified
        if timeout:
            for test in tests:
                # Override timeout calculation
                test._custom_timeout = timeout

        # Connect to ComfyUI
        if verbose:
            print(f"Connecting to ComfyUI at {target}...")

        client = ComfyClient(server_address=target)

        try:
            client.connect()
            if verbose:
                print(f"Connected successfully\n")
        except ConnectionError as e:
            print(f"Error: {e}", file=sys.stderr)
            print(f"\nMake sure ComfyUI is running at {target}", file=sys.stderr)
            sys.exit(1)

        # Run tests
        runner = TestRunner(strategy=SequentialExecutor())

        if not verbose:
            print(f"Running {len(tests)} test(s)...")

        results = runner.run(
            tests=tests,
            client=client,
            verbose=verbose,
            failfast=failfast,
        )

        # Process results
        passed = 0
        failed = 0
        skipped = 0

        for i, result in enumerate(results):
            # Report to formatter
            formatter.test_started(result.test.name, i, len(results))

            if result.status == TestStatus.PASSED:
                formatter.test_passed(result.test.name)
                passed += 1

            elif result.status == TestStatus.SKIPPED:
                formatter.test_skipped(result.test.name, result.skipped_reason or "Unknown")
                skipped += 1

            else:  # FAILED or ERROR
                error_msg = result.error_message or "Unknown error"
                formatter.test_failed(result.test.name, error_msg)
                failed += 1

        # Print summary
        formatter.summary(passed=passed, failed=failed, skipped=skipped, total=len(results))

        # Print detailed failure information if not verbose
        if failed > 0 and not verbose:
            print("Failed tests:")
            for result in results:
                if result.failed:
                    print(f"  - {result.test.name}")
                    if result.error_message:
                        print(f"    {result.error_message}")

        # Close connection
        client.close()

        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
