"""Test execution strategies and result tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from client import ComfyClient, TestExecution
from workflow_parser import WorkflowTest


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test execution."""
    test: WorkflowTest
    status: TestStatus
    execution: TestExecution | None = None
    error_message: str | None = None
    skipped_reason: str | None = None

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED

    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in (TestStatus.FAILED, TestStatus.ERROR)


class ExecutionStrategy(ABC):
    """Abstract base class for test execution strategies."""

    @abstractmethod
    def execute_tests(
        self,
        tests: list[WorkflowTest],
        client: ComfyClient,
        verbose: bool = False,
        failfast: bool = False,
    ) -> list[TestResult]:
        """
        Execute a list of tests.

        Args:
            tests: List of WorkflowTest objects to execute
            client: Connected ComfyUI client
            verbose: Enable verbose output
            failfast: Stop on first failure

        Returns:
            List of TestResult objects
        """
        pass


class SequentialExecutor(ExecutionStrategy):
    """Execute tests sequentially, one at a time."""

    def execute_tests(
        self,
        tests: list[WorkflowTest],
        client: ComfyClient,
        verbose: bool = False,
        failfast: bool = False,
    ) -> list[TestResult]:
        """Execute tests sequentially."""
        results = []

        for test in tests:
            if verbose:
                print(f"\nRunning: {test.name}")

            result = self._execute_single_test(test, client, verbose)
            results.append(result)

            # Stop on first failure if failfast is enabled
            if failfast and result.failed:
                if verbose:
                    print("\nStopping due to failure (--failfast)")
                break

        return results

    def _execute_single_test(
        self,
        test: WorkflowTest,
        client: ComfyClient,
        verbose: bool,
    ) -> TestResult:
        """
        Execute a single test workflow.

        Args:
            test: WorkflowTest to execute
            client: Connected ComfyUI client
            verbose: Enable verbose output

        Returns:
            TestResult with execution outcome
        """
        try:
            # Execute workflow with timeout
            execution = client.execute_workflow(
                workflow=test.workflow,
                timeout=test.timeout,
                verbose=verbose,
            )

            # Check for execution errors
            if execution.has_error and execution.error is not None:
                error_data = execution.error
                error_type = error_data.get('exception_type', 'Unknown')
                error_msg = error_data.get('exception_message', 'No message')
                node_id = error_data.get('node_id', 'unknown')

                return TestResult(
                    test=test,
                    status=TestStatus.ERROR,
                    execution=execution,
                    error_message=f"{error_type} in node {node_id}: {error_msg}",
                )

            # Validate that all TestMustExecute nodes ran or were cached
            not_executed = self._find_not_executed_nodes(test, execution)

            if not_executed:
                node_list = ', '.join(sorted(not_executed))
                return TestResult(
                    test=test,
                    status=TestStatus.FAILED,
                    execution=execution,
                    error_message=f"TestMustExecute nodes did not execute: {node_list}",
                )

            # Test passed
            return TestResult(
                test=test,
                status=TestStatus.PASSED,
                execution=execution,
            )

        except TimeoutError as e:
            return TestResult(
                test=test,
                status=TestStatus.ERROR,
                error_message=str(e),
            )

        except Exception as e:
            return TestResult(
                test=test,
                status=TestStatus.ERROR,
                error_message=f"Unexpected error: {e}",
            )

    def _find_not_executed_nodes(
        self,
        test: WorkflowTest,
        execution: TestExecution,
    ) -> set[str]:
        """
        Find TestMustExecute nodes that were not executed.

        Args:
            test: WorkflowTest being validated
            execution: TestExecution with results

        Returns:
            Set of node IDs that did not execute
        """
        not_executed = set()

        for node_id in test.test_must_execute_nodes:
            if not execution.was_executed(node_id):
                not_executed.add(node_id)

        return not_executed


class TestRunner:
    """Main test runner that coordinates test execution."""

    def __init__(self, strategy: ExecutionStrategy):
        """
        Initialize test runner with an execution strategy.

        Args:
            strategy: Execution strategy to use (Sequential, Parallel, etc.)
        """
        self.strategy = strategy

    def run(
        self,
        tests: list[WorkflowTest],
        client: ComfyClient,
        verbose: bool = False,
        failfast: bool = False,
    ) -> list[TestResult]:
        """
        Run tests using the configured strategy.

        Args:
            tests: List of WorkflowTest objects
            client: Connected ComfyUI client
            verbose: Enable verbose output
            failfast: Stop on first failure

        Returns:
            List of TestResult objects
        """
        return self.strategy.execute_tests(
            tests=tests,
            client=client,
            verbose=verbose,
            failfast=failfast,
        )
