"""Centralized Orchestrator for conversation generation.

This module provides the main controller that coordinates all agents
to generate complete, high-quality conversations.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.agents import (
    ScenarioPlannerAgent,
    UserSimulatorAgent,
    AssistantAgent,
    ToolExecutorAgent,
    JudgeAgent,
    RepairAgent,
    DiversitySteeringAgent,
)
from src.models.context import ConversationContext
from src.sampling.constraints import SamplingConstraints

if TYPE_CHECKING:
    from src.graph.client import GraphClient
    from src.llm import LLMClient
    from src.models.judge_scores import JudgeScores
    from src.sampling.dfs_sampler import DFSSampler


@dataclass
class GenerationResult:
    """Result of a single conversation generation.

    Attributes:
        conversation: The generated conversation context
        success: Whether generation succeeded
        scores: Judge scores if evaluated
        attempts: Number of attempts made
        repaired: Whether repair was applied
        error: Error message if failed
    """
    conversation: Optional[ConversationContext] = None
    success: bool = False
    scores: Optional["JudgeScores"] = None
    attempts: int = 0
    repaired: bool = False
    error: Optional[str] = None


@dataclass
class DatasetResult:
    """Result of dataset generation.

    Attributes:
        conversations: List of successful generation results
        failed_count: Number of failed generations
        total_attempts: Total generation attempts
        generation_time: Total time in seconds
    """
    conversations: List[GenerationResult] = field(default_factory=list)
    failed_count: int = 0
    total_attempts: int = 0
    generation_time: float = 0.0

    @property
    def success_count(self) -> int:
        """Number of successful generations."""
        return len(self.conversations)

    @property
    def success_rate(self) -> float:
        """Ratio of successful to total requested."""
        total = self.success_count + self.failed_count
        return self.success_count / total if total > 0 else 0.0


class ConversationOrchestrator:
    """Main controller for conversation generation.

    Coordinates all agents to generate complete, high-quality
    conversations from sampled tool chains.

    Attributes:
        llm: LLMClient for all agents
        graph: GraphClient for tool graph
        sampler: DFSSampler for tool chain sampling
        quality_threshold: Minimum average score to accept (default 6.0)
        max_retries: Maximum attempts per conversation (default 3)
        use_steering: Enable cross-conversation diversity steering

    Example:
        >>> orchestrator = ConversationOrchestrator(llm, graph, sampler)
        >>> result = orchestrator.generate_dataset(count=100)
        >>> print(f"Generated {result.success_count} conversations")
    """

    def __init__(
        self,
        llm: "LLMClient",
        graph: "GraphClient",
        sampler: "DFSSampler",
        quality_threshold: float = 6.0,
        max_retries: int = 3,
        use_steering: bool = True,
    ) -> None:
        """Initialize the orchestrator with all agents.

        Args:
            llm: LLMClient instance for all agents
            graph: GraphClient for tool graph access
            sampler: DFSSampler for tool chain sampling
            quality_threshold: Minimum average score to accept
            max_retries: Maximum generation attempts
            use_steering: Enable diversity steering
        """
        self.llm = llm
        self.graph = graph
        self.sampler = sampler
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.use_steering = use_steering

        # Initialize all agents
        self.scenario_planner = ScenarioPlannerAgent(
            llm=llm, name="scenario_planner", graph_client=graph
        )
        self.user_simulator = UserSimulatorAgent(llm=llm, name="user_simulator")
        self.assistant = AssistantAgent(llm=llm, name="assistant")
        self.tool_executor = ToolExecutorAgent(llm=llm, name="tool_executor")
        self.judge = JudgeAgent(llm=llm, name="judge")
        self.repair = RepairAgent(llm=llm, name="repair")

        # Initialize diversity steering
        self.steering = DiversitySteeringAgent()

    def generate_dataset(
        self,
        count: int,
        base_seed: Optional[int] = None,
        constraints: Optional[SamplingConstraints] = None,
    ) -> DatasetResult:
        """Generate multiple conversations.

        Args:
            count: Number of conversations to generate
            base_seed: Base random seed (conversation i uses base_seed + i)
            constraints: Optional base sampling constraints

        Returns:
            DatasetResult with all successful conversations
        """
        start_time = time.time()
        result = DatasetResult()

        for i in range(count):
            # Set seed for reproducibility
            seed = (base_seed + i) if base_seed is not None else None

            # Get steering constraints if enabled
            if self.use_steering and constraints is None:
                current_constraints = self.steering.suggest_constraints()
            elif constraints is not None:
                current_constraints = constraints
            else:
                current_constraints = SamplingConstraints()

            # Generate single conversation
            gen_result = self.generate_single(
                seed=seed,
                constraints=current_constraints,
            )

            result.total_attempts += gen_result.attempts

            if gen_result.success:
                result.conversations.append(gen_result)

                # Record for diversity tracking
                if self.use_steering and gen_result.conversation:
                    self.steering.record(gen_result.conversation)
            else:
                result.failed_count += 1

        result.generation_time = time.time() - start_time
        return result

    def generate_single(
        self,
        seed: Optional[int] = None,
        constraints: Optional[SamplingConstraints] = None,
    ) -> GenerationResult:
        """Generate a single conversation with retry logic.

        Attempts to generate a high-quality conversation, with
        escalating recovery strategies on failure:
        1. Regenerate with same tool chain
        2. Resample new tool chain
        3. Discard and report failure

        Args:
            seed: Random seed for reproducibility
            constraints: Sampling constraints

        Returns:
            GenerationResult with conversation or error
        """
        if seed is not None:
            random.seed(seed)

        constraints = constraints or SamplingConstraints()
        result = GenerationResult()
        current_tool_chain: Optional[List[str]] = None

        for attempt in range(self.max_retries):
            result.attempts = attempt + 1

            try:
                # Sample tool chain (resample on attempt > 0 if previous failed badly)
                if current_tool_chain is None or attempt > 0:
                    # Get diversity weights if steering enabled
                    if self.use_steering:
                        diversity_weights = self.steering.get_diversity_weights()
                        self.sampler.diversity_weights = diversity_weights

                    current_tool_chain = self.sampler.sample()

                    if not current_tool_chain:
                        result.error = "Failed to sample tool chain"
                        continue

                # Create context
                # Extract endpoint IDs for tool_chain, store full data in grounding_values
                endpoint_ids = [ep["endpoint_id"] for ep in current_tool_chain]
                context = ConversationContext(
                    tool_chain=endpoint_ids,
                    target_steps=len(current_tool_chain),
                    seed=seed,
                )
                # Store full sampled chain data for scenario planner to use
                context.grounding_values["sampled_chain_data"] = current_tool_chain

                # Generate scenario
                context = self.scenario_planner.generate(context)

                # Generate conversation turns
                context = self._generate_conversation(context)

                # Judge quality
                context = self.judge.generate(context)
                scores = context.grounding_values.get("judge_scores")

                if scores and scores.average >= self.quality_threshold:
                    # Success!
                    result.conversation = context
                    result.success = True
                    result.scores = scores
                    return result

                # Quality too low, try repair
                if scores and scores.average < self.quality_threshold:
                    repaired_context = self.repair.generate(context)

                    # Re-evaluate after repair
                    repaired_context = self.judge.generate(repaired_context)
                    repaired_scores = repaired_context.grounding_values.get("judge_scores")

                    if repaired_scores and repaired_scores.average >= self.quality_threshold:
                        result.conversation = repaired_context
                        result.success = True
                        result.scores = repaired_scores
                        result.repaired = True
                        return result

                    # Still not good enough, will retry with new chain
                    result.error = f"Quality too low after repair: {repaired_scores.average if repaired_scores else 'N/A'}"

            except Exception as e:
                result.error = str(e)
                # Reset tool chain to force resample on next attempt
                current_tool_chain = None

        # All retries exhausted
        return result

    def _generate_conversation(
        self,
        context: ConversationContext,
    ) -> ConversationContext:
        """Generate conversation turns until complete.

        Alternates between user and assistant messages,
        executing tool calls and extracting grounding values.

        Args:
            context: Initialized context with scenario

        Returns:
            Completed context with all messages
        """
        max_turns = 20  # Safety limit

        for turn in range(max_turns):
            if context.is_complete:
                break

            # Generate user message
            context = self.user_simulator.generate(context)

            # Generate assistant response
            context = self.assistant.generate(context)

            # Check for tool calls in last assistant message
            last_message = context.messages[-1] if context.messages else None

            if last_message and last_message.role == "assistant" and last_message.tool_calls:
                # Execute each tool call
                for tool_call in last_message.tool_calls:
                    # Execute tool (also extracts grounding values internally)
                    context = self.tool_executor.generate(context)

        return context

    def register_endpoints(self, endpoints: Dict[str, Any]) -> None:
        """Register endpoints with the diversity steering agent.

        Args:
            endpoints: Dict mapping endpoint_id to Endpoint objects
        """
        for endpoint_id, endpoint in endpoints.items():
            tool_id = getattr(endpoint, 'tool_id', endpoint_id)
            domain = getattr(endpoint, 'domain', None)
            self.steering.register_endpoint(endpoint_id, tool_id, domain)

    def get_diversity_metrics(self):
        """Get current diversity metrics.

        Returns:
            DiversityMetrics from the steering agent
        """
        return self.steering.compute_metrics()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConversationOrchestrator("
            f"threshold={self.quality_threshold}, "
            f"max_retries={self.max_retries}, "
            f"steering={self.use_steering})"
        )
