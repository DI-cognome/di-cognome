"""
Decision recording and bond creation.

This module handles the output side of cognition:
- Recording decisions with rationale
- Creating new bonds from decisions
- Building decision chains (causal traces)
- Persisting decisions to memory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence
from pathlib import Path
import json
import hashlib

from ..core.token_id import TokenID
from ..core.pair_bond import PairBondMap, PairBond
from .context import ActivatedBond


@dataclass
class Decision:
    """
    A recorded decision with full context.

    Decisions create new bonds:
    - Input bonds (context used)
    - Output bonds (result of decision)
    - Causal bond (this decision -> outcome)
    """
    id: str
    timestamp: datetime
    agent: str
    action: str
    rationale: str
    input_context: list[ActivatedBond]
    output_tokens: list[TokenID]
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            content = f"{self.agent}:{self.action}:{self.timestamp.isoformat()}"
            self.id = hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def decision_bond(self) -> PairBond | None:
        """Create a bond representing this decision (input -> output)."""
        if not self.input_context or not self.output_tokens:
            return None

        # Use highest-activation input token and first output token
        top_input = max(self.input_context, key=lambda b: b.activation)
        input_token = top_input.bond.left  # Start of most relevant bond

        return PairBond(input_token, self.output_tokens[0])

    def to_pbm(self) -> PairBondMap:
        """Convert decision to a PBM for storage/merging."""
        pbm = PairBondMap()

        # Add output sequence bonds
        if len(self.output_tokens) > 1:
            pbm.add_sequence(self.output_tokens)

        # Add decision bond (input -> output)
        decision_bond = self.decision_bond
        if decision_bond:
            pbm.add_bond(decision_bond.left, decision_bond.right)

        return pbm

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'agent': self.agent,
            'action': self.action,
            'rationale': self.rationale,
            'input_context': [
                {
                    'bond': str(ab.bond),
                    'activation': ab.activation,
                    'path_length': ab.path_length,
                }
                for ab in self.input_context
            ],
            'output_tokens': [t.to_string() for t in self.output_tokens],
            'confidence': self.confidence,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Decision:
        """Deserialize from storage."""
        # Note: input_context loses full bond structure, just metadata
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            agent=data['agent'],
            action=data['action'],
            rationale=data['rationale'],
            input_context=[],  # Cannot fully reconstruct
            output_tokens=[TokenID.from_string(t) for t in data.get('output_tokens', [])],
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class DecisionChain:
    """
    A chain of related decisions (causal trace).

    Tracks how one decision led to another.
    """
    decisions: list[Decision] = field(default_factory=list)
    root_context: str = ""  # What started this chain

    def add(self, decision: Decision) -> None:
        """Add a decision to the chain."""
        self.decisions.append(decision)

    @property
    def length(self) -> int:
        return len(self.decisions)

    @property
    def total_confidence(self) -> float:
        """Compound confidence (product of all decisions)."""
        if not self.decisions:
            return 0.0
        conf = 1.0
        for d in self.decisions:
            conf *= d.confidence
        return conf

    def to_pbm(self) -> PairBondMap:
        """Convert entire chain to PBM."""
        pbm = PairBondMap()
        for decision in self.decisions:
            pbm.merge(decision.to_pbm())
        return pbm


class DecisionRecorder:
    """
    Records and stores decisions.

    Maintains:
    - Decision history
    - Decision chains
    - Integration with knowledge base
    """

    def __init__(
        self,
        agent_name: str,
        storage_path: Path | str | None = None,
        knowledge_pbm: PairBondMap | None = None,
    ) -> None:
        """
        Initialize decision recorder.

        Args:
            agent_name: Name of the agent making decisions
            storage_path: Optional path for persistent storage
            knowledge_pbm: Knowledge base to update with decisions
        """
        self.agent_name = agent_name
        self.storage_path = Path(storage_path) if storage_path else None
        self.knowledge_pbm = knowledge_pbm or PairBondMap()
        self._history: list[Decision] = []
        self._chains: dict[str, DecisionChain] = {}
        self._current_chain: DecisionChain | None = None

    def record(
        self,
        action: str,
        rationale: str,
        input_context: list[ActivatedBond],
        output: str,
        confidence: float = 1.0,
        chain_id: str | None = None,
        **metadata,
    ) -> Decision:
        """
        Record a decision.

        Args:
            action: What was decided/done
            rationale: Why this decision was made
            input_context: Relevant context used
            output: Result of the decision
            confidence: How confident in this decision (0-1)
            chain_id: Optional chain to add this decision to
            **metadata: Additional metadata

        Returns:
            The recorded Decision
        """
        # Tokenize output
        output_tokens = [TokenID.byte(b) for b in output.encode('utf-8')]

        decision = Decision(
            id="",  # Will be generated
            timestamp=datetime.now(timezone.utc),
            agent=self.agent_name,
            action=action,
            rationale=rationale,
            input_context=input_context,
            output_tokens=output_tokens,
            confidence=confidence,
            metadata=metadata,
        )

        # Add to history
        self._history.append(decision)

        # Add to chain if specified
        if chain_id:
            if chain_id not in self._chains:
                self._chains[chain_id] = DecisionChain(root_context=action)
            self._chains[chain_id].add(decision)

        # Update knowledge base with new bonds
        self.knowledge_pbm.merge(decision.to_pbm())

        # Persist if configured
        if self.storage_path:
            self._persist_decision(decision)

        return decision

    def start_chain(self, context: str) -> str:
        """Start a new decision chain, returns chain ID."""
        chain_id = hashlib.sha256(
            f"{self.agent_name}:{context}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:8]

        self._chains[chain_id] = DecisionChain(root_context=context)
        self._current_chain = self._chains[chain_id]

        return chain_id

    def get_chain(self, chain_id: str) -> DecisionChain | None:
        """Get a decision chain by ID."""
        return self._chains.get(chain_id)

    def recent_decisions(self, n: int = 10) -> list[Decision]:
        """Get N most recent decisions."""
        return self._history[-n:]

    def decisions_by_action(self, action_pattern: str) -> list[Decision]:
        """Find decisions matching an action pattern."""
        return [d for d in self._history if action_pattern in d.action]

    def _persist_decision(self, decision: Decision) -> None:
        """Save decision to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Daily file for decisions
        date_str = decision.timestamp.strftime('%Y-%m-%d')
        path = self.storage_path / f'decisions-{date_str}.jsonl'

        with open(path, 'a') as f:
            f.write(json.dumps(decision.to_dict()) + '\n')

    def load_history(self, days: int = 7) -> None:
        """Load decision history from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        for path in sorted(self.storage_path.glob('decisions-*.jsonl'))[-days:]:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        decision = Decision.from_dict(data)
                        self._history.append(decision)


def record_decision(
    agent: str,
    action: str,
    rationale: str,
    output: str,
    context: list[ActivatedBond] | None = None,
    confidence: float = 1.0,
) -> Decision:
    """
    Quick interface to record a decision.

    For simple use cases without managing a full recorder.
    """
    recorder = DecisionRecorder(agent)
    return recorder.record(
        action=action,
        rationale=rationale,
        input_context=context or [],
        output=output,
        confidence=confidence,
    )
