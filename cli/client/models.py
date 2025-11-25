"""Additional models not covered by OpenAPI spec."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestExecution:
    """Tracks execution state of a workflow."""

    prompt_id: str
    runs: set[str] = field(default_factory=set)
    cached: set[str] = field(default_factory=set)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def did_run(self, node_id: str) -> bool:
        return node_id in self.runs

    def was_cached(self, node_id: str) -> bool:
        return node_id in self.cached

    def was_executed(self, node_id: str) -> bool:
        return self.did_run(node_id) or self.was_cached(node_id)

    @property
    def has_error(self) -> bool:
        return self.error is not None
