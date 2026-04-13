"""Pattern classes for tool chain execution.

This module provides classes representing different execution patterns
for tool chains, including sequential, parallel, branching, and iterative
patterns. Each pattern can be converted to an execution plan.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolStep:
    """Represents a single step in an execution plan.

    Attributes:
        endpoint_id: The endpoint ID for this step
        depends_on: List of endpoint IDs that must complete before this step
        is_parallel: Whether this step can run in parallel with others
    """

    endpoint_id: str
    depends_on: List[str] = field(default_factory=list)
    is_parallel: bool = False


class BaseChainPattern(ABC):
    """Abstract base class for all chain patterns.

    All pattern classes must inherit from this base class and implement
    the required abstract methods.
    """

    @property
    @abstractmethod
    def pattern_type(self) -> str:
        """Return the pattern type name.

        Returns:
            Pattern type string (e.g., 'sequential', 'parallel')
        """
        pass

    @abstractmethod
    def get_endpoints(self) -> List[str]:
        """Return all endpoint IDs in this pattern.

        Returns:
            List of all endpoint IDs that are part of this pattern
        """
        pass

    @abstractmethod
    def to_execution_plan(self, **kwargs) -> List[ToolStep]:
        """Convert pattern to a list of ToolSteps for execution.

        Returns:
            List of ToolStep objects representing the execution order
        """
        pass


class SequentialChain(BaseChainPattern):
    """Linear chain pattern: A → B → C

    Each step depends on the previous step completing before it can start.

    Attributes:
        steps: List of endpoint IDs in execution order

    Example:
        >>> chain = SequentialChain(steps=["ep_a", "ep_b", "ep_c"])
        >>> plan = chain.to_execution_plan()
        >>> # ep_a runs first, ep_b depends on ep_a, ep_c depends on ep_b
    """

    def __init__(self, steps: List[str]) -> None:
        """Initialize a sequential chain.

        Args:
            steps: List of endpoint IDs in execution order
        """
        self.steps = steps

    @property
    def pattern_type(self) -> str:
        """Return 'sequential' as the pattern type."""
        return "sequential"

    def get_endpoints(self) -> List[str]:
        """Return all endpoint IDs in the chain."""
        return list(self.steps)

    def to_execution_plan(self, **kwargs) -> List[ToolStep]:
        """Convert to execution plan with sequential dependencies.

        Each step depends on the previous step. The first step has no
        dependencies.

        Returns:
            List of ToolStep objects with sequential dependencies
        """
        plan = []

        for i, endpoint_id in enumerate(self.steps):
            if i == 0:
                # First step has no dependencies
                depends_on = []
            else:
                # Each subsequent step depends on the previous one
                depends_on = [self.steps[i - 1]]

            plan.append(ToolStep(
                endpoint_id=endpoint_id,
                depends_on=depends_on,
                is_parallel=False,
            ))

        return plan


class ParallelChain(BaseChainPattern):
    """Parallel execution pattern: [A, B] → C → D

    Some steps run in parallel, then sequential steps follow after all
    parallel steps complete.

    Attributes:
        parallel_steps: List of endpoint IDs to run in parallel
        then_steps: List of endpoint IDs to run sequentially after parallel completion

    Example:
        >>> chain = ParallelChain(
        ...     parallel_steps=["ep_a", "ep_b"],
        ...     then_steps=["ep_c"]
        ... )
        >>> plan = chain.to_execution_plan()
        >>> # ep_a and ep_b run in parallel, ep_c runs after both complete
    """

    def __init__(self, parallel_steps: List[str], then_steps: List[str]) -> None:
        """Initialize a parallel chain.

        Args:
            parallel_steps: List of endpoint IDs to run in parallel
            then_steps: List of endpoint IDs to run sequentially after parallel completion
        """
        self.parallel_steps = parallel_steps
        self.then_steps = then_steps

    @property
    def pattern_type(self) -> str:
        """Return 'parallel' as the pattern type."""
        return "parallel"

    def get_endpoints(self) -> List[str]:
        """Return all endpoint IDs in the chain."""
        return self.parallel_steps + self.then_steps

    def to_execution_plan(self, **kwargs) -> List[ToolStep]:
        """Convert to execution plan with parallel and sequential steps.

        Parallel steps have no dependencies and is_parallel=True.
        The first then_step depends on ALL parallel_steps.
        Subsequent then_steps depend on the previous then_step.

        Returns:
            List of ToolStep objects with appropriate dependencies
        """
        plan = []

        # Add parallel steps (no dependencies, is_parallel=True)
        for endpoint_id in self.parallel_steps:
            plan.append(ToolStep(
                endpoint_id=endpoint_id,
                depends_on=[],
                is_parallel=True,
            ))

        # Add then_steps with appropriate dependencies
        for i, endpoint_id in enumerate(self.then_steps):
            if i == 0:
                # First then_step depends on ALL parallel steps
                depends_on = list(self.parallel_steps)
            else:
                # Subsequent then_steps depend on the previous then_step
                depends_on = [self.then_steps[i - 1]]

            plan.append(ToolStep(
                endpoint_id=endpoint_id,
                depends_on=depends_on,
                is_parallel=False,
            ))

        return plan


class BranchingChain(BaseChainPattern):
    """Conditional branching pattern: A → [B or C] → D

    Execution takes one of multiple branches based on a condition,
    then converges to a merge step.

    Attributes:
        start: Initial endpoint ID
        branches: Dict mapping condition names to lists of endpoint IDs
        merge: Final endpoint ID after branch convergence

    Example:
        >>> chain = BranchingChain(
        ...     start="check_weather",
        ...     branches={
        ...         "sunny": ["outdoor_plan"],
        ...         "rainy": ["indoor_plan"],
        ...     },
        ...     merge="execute_plan"
        ... )
        >>> plan = chain.to_execution_plan(selected_branch="sunny")
    """

    def __init__(
        self,
        start: str,
        branches: Dict[str, List[str]],
        merge: str,
    ) -> None:
        """Initialize a branching chain.

        Args:
            start: Initial endpoint ID
            branches: Dict mapping condition names to lists of endpoint IDs
            merge: Final endpoint ID after branch convergence
        """
        self.start = start
        self.branches = branches
        self.merge = merge

    @property
    def pattern_type(self) -> str:
        """Return 'branching' as the pattern type."""
        return "branching"

    def get_endpoints(self) -> List[str]:
        """Return all endpoint IDs in the chain (including all branches)."""
        endpoints = [self.start]
        for branch_steps in self.branches.values():
            endpoints.extend(branch_steps)
        endpoints.append(self.merge)
        return endpoints

    def select_branch(self, condition: str) -> List[str]:
        """Select which branch to execute based on condition.

        Args:
            condition: The condition key to select

        Returns:
            List of endpoint IDs for the selected branch, or empty list if not found
        """
        return self.branches.get(condition, [])

    def to_execution_plan(
        self,
        selected_branch: Optional[str] = None,
        **kwargs,
    ) -> List[ToolStep]:
        """Convert to execution plan with selected branch.

        Args:
            selected_branch: The branch condition to execute. If None,
                           picks the first available branch.

        Returns:
            List of ToolStep objects for the selected execution path
        """
        plan = []

        # Start step has no dependencies
        plan.append(ToolStep(
            endpoint_id=self.start,
            depends_on=[],
            is_parallel=False,
        ))

        # Select branch (default to first if not specified)
        if selected_branch is None:
            selected_branch = next(iter(self.branches.keys()), None)

        # Get branch steps
        if selected_branch is None or selected_branch not in self.branches:
            # No valid branch - just connect start to merge
            branch_steps = []
        else:
            branch_steps = self.branches[selected_branch]

        # Add branch steps (sequential within branch, first depends on start)
        prev_step = self.start
        for endpoint_id in branch_steps:
            plan.append(ToolStep(
                endpoint_id=endpoint_id,
                depends_on=[prev_step],
                is_parallel=False,
            ))
            prev_step = endpoint_id

        # Merge step depends on last step (either last branch step or start)
        plan.append(ToolStep(
            endpoint_id=self.merge,
            depends_on=[prev_step],
            is_parallel=False,
        ))

        return plan


class IterativeChain(BaseChainPattern):
    """Iterative pattern: A → B → B → B → C

    A step is repeated multiple times before proceeding to the end step.

    Attributes:
        start: Initial endpoint ID
        loop_step: Endpoint ID to repeat
        loop_count: Number of times to repeat the loop step
        end: Final endpoint ID after loop

    Example:
        >>> chain = IterativeChain(
        ...     start="init",
        ...     loop_step="process_batch",
        ...     loop_count=3,
        ...     end="finalize"
        ... )
        >>> plan = chain.to_execution_plan()
        >>> # init → process_batch → process_batch → process_batch → finalize
    """

    def __init__(
        self,
        start: str,
        loop_step: str,
        loop_count: int,
        end: str,
    ) -> None:
        """Initialize an iterative chain.

        Args:
            start: Initial endpoint ID
            loop_step: Endpoint ID to repeat
            loop_count: Number of times to repeat the loop step
            end: Final endpoint ID after loop
        """
        self.start = start
        self.loop_step = loop_step
        self.loop_count = loop_count
        self.end = end

    @property
    def pattern_type(self) -> str:
        """Return 'iterative' as the pattern type."""
        return "iterative"

    def get_endpoints(self) -> List[str]:
        """Return unique endpoint IDs in the chain."""
        return [self.start, self.loop_step, self.end]

    def to_execution_plan(self, **kwargs) -> List[ToolStep]:
        """Convert to execution plan with repeated loop steps.

        The start step has no dependencies. Each loop iteration depends
        on the previous step. The end step depends on the last loop iteration.

        Returns:
            List of ToolStep objects with loop iterations expanded
        """
        plan = []

        # Start step has no dependencies
        plan.append(ToolStep(
            endpoint_id=self.start,
            depends_on=[],
            is_parallel=False,
        ))

        # Track previous step for dependencies
        prev_step = self.start

        # Add loop iterations
        for _ in range(self.loop_count):
            plan.append(ToolStep(
                endpoint_id=self.loop_step,
                depends_on=[prev_step],
                is_parallel=False,
            ))
            prev_step = self.loop_step

        # End step depends on last loop iteration (or start if loop_count=0)
        plan.append(ToolStep(
            endpoint_id=self.end,
            depends_on=[prev_step],
            is_parallel=False,
        ))

        return plan
