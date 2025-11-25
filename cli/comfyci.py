#!/usr/bin/env -S uv run
"""ComfyCI - CLI tool for running ComfyUI test workflows."""

import subprocess
import sys

import click

from client import ServerUnreachableError, create_client
from formatters import SimpleFormatter
from test_runner import SequentialExecutor, TestRunner, TestStatus
from workflow_parser import discover_tests


def get_op_secret(op_exe: str, entry: str, field: str) -> str:
    """Fetch a secret from 1Password CLI."""
    ref = f"op://{entry}/{field}"
    result = subprocess.run(
        [op_exe, "read", ref],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@click.command()
@click.argument('patterns', nargs=-1, required=True)
@click.option(
    '--server',
    default='localhost:8188',
    envvar='COMFY_SERVER',
    help='ComfyUI server address (host:port)',
    show_default=True,
)
@click.option(
    '--ssl',
    is_flag=True,
    help='Use HTTPS/WSS for server connection',
)
@click.option(
    '--cloud',
    is_flag=True,
    help='Use Comfy Cloud API (e.g., /jobs instead of /history)',
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
@click.option(
    '--op-exe',
    default='op.exe',
    help='1Password CLI executable (op or op.exe)',
    show_default=True,
)
@click.option(
    '--op-entry',
    default=None,
    help='1Password entry path (e.g., Employee/TestCI). Fetches email, password, FIREBASE_API_KEY.',
)
def main(
    patterns: tuple,
    server: str,
    ssl: bool,
    cloud: bool,
    cpu: bool,
    verbose: bool,
    failfast: bool,
    timeout: int | None,
    no_color: bool,
    op_exe: str,
    op_entry: str | None,
):
    """
    Run ComfyUI test workflows.

    PATTERNS: Glob patterns for test workflow JSON files (e.g., ./tests/**/*.json)

    Examples:

        comfyci ./tests/**/*.json --server localhost:8188

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
            print(f"Connecting to ComfyUI at {server}...")

        # Fetch credentials from 1Password if --op-entry is provided
        if op_entry:
            if verbose:
                print(f"Fetching credentials from 1Password ({op_entry})...")
            try:
                email = get_op_secret(op_exe, op_entry, "email")
                password = get_op_secret(op_exe, op_entry, "password")
                api_key = get_op_secret(op_exe, op_entry, "FIREBASE_API_KEY")
                client = create_client(
                    server,
                    use_ssl=ssl,
                    cloud=cloud,
                    email=email,
                    password=password,
                    api_key=api_key,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error fetching credentials from 1Password: {e.stderr}", file=sys.stderr)
                print(f"Make sure you're signed in to 1Password ({op_exe})", file=sys.stderr)
                sys.exit(1)
        else:
            client = create_client(server, use_ssl=ssl, cloud=cloud)

        try:
            client.connect()
            if verbose:
                print("Connected successfully\n")
        except ServerUnreachableError as e:
            print(f"Error: {e}", file=sys.stderr)
            print(f"\nMake sure ComfyUI is running at {server}", file=sys.stderr)
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
