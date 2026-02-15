"""
Reasoning: Inference layer connecting context, generation, and decisions.

This module handles the "what to do" question:
1. Query → context retrieval (what's relevant)
2. Context → generation (what to say/do)
3. Response → decision recording (what was decided)

Physics provides the "what's connected" - reasoning provides "what to do about it."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
from datetime import datetime

from .bridge import OllamaBridge, Bond, GenerationConfig, GenerationResult
from .context import ContextRetriever, ActivatedBond
from .decisions import DecisionRecorder, Decision
from .identity import IdentitySeed, IdentityFilter


@dataclass
class ReasoningConfig:
    """Configuration for reasoning engine."""
    max_context_bonds: int = 20
    min_activation: float = 0.1
    # Default model - tinyllama for testing (fast, small).
    # Use llama3, mistral, etc. for better quality in production.
    model: str = "tinyllama:latest"
    record_decisions: bool = True
    identity_token: str | None = None


@dataclass
class ReasoningResult:
    """Result of a reasoning step."""
    query: str
    response: str
    context_bonds: list[ActivatedBond]
    decision: Decision | None
    generation_result: GenerationResult
    duration_ms: float
    timestamp: str

    def summary(self) -> str:
        """One-line summary of reasoning result."""
        return (
            f"Q: {self.query[:30]}... → "
            f"R: {self.response[:30]}... "
            f"({len(self.context_bonds)} bonds, {self.duration_ms:.0f}ms)"
        )


class ReasoningEngine:
    """
    Main reasoning engine connecting all cognition components.

    Flow:
        query → context retrieval → LLM generation → decision recording

    The engine orchestrates but doesn't do the heavy lifting:
    - Context retrieval uses physics-based bond traversal
    - Generation uses LLM via bridge
    - Decisions are recorded for memory
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        context_retriever: ContextRetriever | None = None,
        bridge: OllamaBridge | None = None,
        decision_recorder: DecisionRecorder | None = None,
        identity: IdentitySeed | None = None,
    ) -> None:
        """
        Initialize reasoning engine.

        Args:
            config: Reasoning configuration
            context_retriever: For finding relevant bonds
            bridge: For LLM generation
            decision_recorder: For recording decisions
            identity: Identity seed for filtering
        """
        self.config = config or ReasoningConfig()

        # Create default context retriever with empty PBM if not provided
        if context_retriever is None:
            from ..core.pair_bond import PairBondMap
            context_retriever = ContextRetriever(PairBondMap())
        self.context = context_retriever

        self.bridge = bridge or OllamaBridge(
            GenerationConfig(model=self.config.model)
        )

        # Create default decision recorder with agent name from identity_token
        if decision_recorder is None:
            agent_name = self.config.identity_token or "reasoning_engine"
            decision_recorder = DecisionRecorder(agent_name)
        self.decisions = decision_recorder

        self.identity = identity

    def reason(
        self,
        query: str,
        additional_context: list[Bond] | None = None,
        decision_type: str = "inference",
    ) -> ReasoningResult:
        """
        Perform a reasoning step.

        Args:
            query: The question or task
            additional_context: Extra bonds to include
            decision_type: Type of decision being made

        Returns:
            ReasoningResult with response and metadata
        """
        start_time = datetime.utcnow()

        # Step 1: Retrieve relevant context
        from ..core.token_id import TokenID
        query_tokens = [TokenID.byte(b) for b in query.encode('utf-8')]
        context_result = self.context.get_context(
            query_tokens=query_tokens,
            max_bonds=self.config.max_context_bonds,
            min_activation=self.config.min_activation,
        )

        # Step 2: Convert to bridge Bond format
        bridge_bonds = self._convert_bonds(context_result.activated_bonds)
        if additional_context:
            bridge_bonds.extend(additional_context)

        # Step 3: Generate response via LLM
        gen_result = self.bridge.generate(
            query=query,
            context_bonds=bridge_bonds,
            identity_token=self.config.identity_token,
        )

        # Step 4: Record decision if enabled
        decision = None
        if self.config.record_decisions:
            decision = self.decisions.record(
                query=query,
                response=gen_result.response,
                context_bonds=len(bridge_bonds),
                decision_type=decision_type,
                identity_token=self.config.identity_token,
            )

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ReasoningResult(
            query=query,
            response=gen_result.response,
            context_bonds=context_result.activated_bonds,
            decision=decision,
            generation_result=gen_result,
            duration_ms=duration,
            timestamp=datetime.utcnow().isoformat(),
        )

    def reason_with_callback(
        self,
        query: str,
        on_token: Callable[[str], None],
        additional_context: list[Bond] | None = None,
    ) -> ReasoningResult:
        """
        Reasoning with streaming response.

        Args:
            query: The question or task
            on_token: Callback for each generated token
            additional_context: Extra bonds to include

        Returns:
            ReasoningResult with full response
        """
        start_time = datetime.utcnow()

        # Get context
        from ..core.token_id import TokenID
        query_tokens = [TokenID.byte(b) for b in query.encode('utf-8')]
        context_result = self.context.get_context(
            query_tokens=query_tokens,
            max_bonds=self.config.max_context_bonds,
            min_activation=self.config.min_activation,
        )

        bridge_bonds = self._convert_bonds(context_result.activated_bonds)
        if additional_context:
            bridge_bonds.extend(additional_context)

        # Stream generation
        gen_result = self.bridge.generate_stream(
            query=query,
            context_bonds=bridge_bonds,
            identity_token=self.config.identity_token,
            callback=on_token,
        )

        # Record decision
        decision = None
        if self.config.record_decisions:
            decision = self.decisions.record(
                query=query,
                response=gen_result.response,
                context_bonds=len(bridge_bonds),
                decision_type="inference",
                identity_token=self.config.identity_token,
            )

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ReasoningResult(
            query=query,
            response=gen_result.response,
            context_bonds=context_result.activated_bonds,
            decision=decision,
            generation_result=gen_result,
            duration_ms=duration,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _convert_bonds(self, activated_bonds: list[ActivatedBond]) -> list[Bond]:
        """Convert ActivatedBond to bridge Bond format."""
        result = []
        for ab in activated_bonds:
            # Extract left/right from the pair bond
            bond = Bond(
                left=str(ab.bond.left) if hasattr(ab.bond, 'left') else str(ab.bond),
                right=str(ab.bond.right) if hasattr(ab.bond, 'right') else "",
                strength=ab.activation,
                metadata={"path_length": ab.path_length},
            )
            result.append(bond)
        return result


# Convenience function
def reason(
    query: str,
    identity_token: str | None = None,
    model: str = "tinyllama:latest",
) -> str:
    """
    Quick reasoning interface.

    Args:
        query: What to reason about
        identity_token: Agent identity
        model: LLM model to use

    Returns:
        Response string
    """
    config = ReasoningConfig(model=model, identity_token=identity_token)
    engine = ReasoningEngine(config)
    result = engine.reason(query)
    return result.response


# CLI for testing
if __name__ == "__main__":
    import sys

    print("Reasoning Engine CLI")
    print("-" * 40)

    config = ReasoningConfig(
        model="tinyllama:latest",
        identity_token="dA.AA.AA.AA",
        record_decisions=False,  # Don't record in test
    )

    engine = ReasoningEngine(config)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What should I do next?"

    print(f"Query: {query}")
    print("Reasoning...")

    try:
        result = engine.reason(query)
        print(f"\nResponse: {result.response}")
        print(f"\nContext bonds: {len(result.context_bonds)}")
        print(f"Duration: {result.duration_ms:.0f}ms")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
