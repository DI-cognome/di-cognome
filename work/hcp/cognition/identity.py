"""
Identity seeds for cognition.

An identity seed encodes what matters to a specific agent:
- Which concepts are weighted more heavily
- What patterns of thought are preferred
- How context gets filtered through "self"

The seed influences cognition without determining it:
- Biases bond formation and decay
- Filters relevant context
- Weights reasoning patterns
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import json

from ..core.token_id import TokenID
from ..core.pair_bond import PairBondMap, create_pbm_from_text
from .context import IdentityFilter


@dataclass
class IdentitySeed:
    """
    An identity seed that influences cognition.

    The seed contains:
    - name: Human-readable identifier
    - core_concepts: Key tokens/concepts central to this identity
    - pattern_weights: How strongly to weight different patterns
    - seed_pbm: Structural representation (bonds that define this identity)
    """
    name: str
    core_concepts: list[str] = field(default_factory=list)
    pattern_weights: dict[str, float] = field(default_factory=dict)
    seed_pbm: PairBondMap | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Build seed PBM from core concepts if not provided
        if self.seed_pbm is None and self.core_concepts:
            self.seed_pbm = PairBondMap()
            for concept in self.core_concepts:
                concept_pbm = create_pbm_from_text(concept)
                self.seed_pbm.merge(concept_pbm)

    def to_identity_filter(self, boost_factor: float = 2.0) -> IdentityFilter:
        """Create an IdentityFilter from this seed."""
        if self.seed_pbm:
            return IdentityFilter.from_seed_pbm(self.seed_pbm, boost_factor)
        return IdentityFilter()

    def get_token_weight(self, token: TokenID) -> float:
        """Get weight for a specific token based on this identity."""
        if self.seed_pbm is None:
            return 1.0

        # Check if token appears in seed
        forward = self.seed_pbm.get_forward_bonds(token)
        backward = self.seed_pbm.get_backward_bonds(token)

        if not forward and not backward:
            return 1.0  # Neutral weight for unknown tokens

        # Weight based on bond frequency in seed
        total = sum(r.count for r in forward.values()) + sum(r.count for r in backward.values())
        if self.seed_pbm.total_bonds > 0:
            return 1.0 + (total / self.seed_pbm.total_bonds) * 2.0

        return 1.0

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            'name': self.name,
            'core_concepts': self.core_concepts,
            'pattern_weights': self.pattern_weights,
            'seed_pbm': self.seed_pbm.to_dict() if self.seed_pbm else None,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IdentitySeed:
        """Deserialize from dictionary."""
        seed_pbm = None
        if data.get('seed_pbm'):
            seed_pbm = PairBondMap.from_dict(data['seed_pbm'])

        return cls(
            name=data['name'],
            core_concepts=data.get('core_concepts', []),
            pattern_weights=data.get('pattern_weights', {}),
            seed_pbm=seed_pbm,
            metadata=data.get('metadata', {}),
        )


class IdentityStore:
    """
    Storage and retrieval of identity seeds.

    Seeds can be:
    - Loaded from files
    - Cached in memory
    - Shared across sessions
    """

    def __init__(self, storage_path: Path | str | None = None) -> None:
        """
        Initialize identity store.

        Args:
            storage_path: Directory for persistent storage (optional)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._cache: dict[str, IdentitySeed] = {}

    def get(self, name: str) -> IdentitySeed | None:
        """Get an identity seed by name."""
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Try loading from storage
        if self.storage_path:
            seed = self._load_from_file(name)
            if seed:
                self._cache[name] = seed
                return seed

        return None

    def put(self, seed: IdentitySeed) -> None:
        """Store an identity seed."""
        self._cache[seed.name] = seed

        # Persist if storage path configured
        if self.storage_path:
            self._save_to_file(seed)

    def list_seeds(self) -> list[str]:
        """List all available seed names."""
        names = set(self._cache.keys())

        if self.storage_path and self.storage_path.exists():
            for path in self.storage_path.glob('*.seed.json'):
                names.add(path.stem.replace('.seed', ''))

        return sorted(names)

    def _load_from_file(self, name: str) -> IdentitySeed | None:
        """Load seed from file."""
        if not self.storage_path:
            return None

        path = self.storage_path / f'{name}.seed.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return IdentitySeed.from_dict(data)

        return None

    def _save_to_file(self, seed: IdentitySeed) -> None:
        """Save seed to file."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        path = self.storage_path / f'{seed.name}.seed.json'

        with open(path, 'w') as f:
            json.dump(seed.to_dict(), f, indent=2)


def create_agent_seed(
    name: str,
    description: str,
    core_concepts: list[str],
    **metadata,
) -> IdentitySeed:
    """
    Convenience function to create an agent identity seed.

    Args:
        name: Agent name (e.g., "planner", "silas")
        description: What this agent does/cares about
        core_concepts: Key concepts that define this agent
        **metadata: Additional metadata

    Returns:
        IdentitySeed configured for this agent
    """
    # Build seed from description + concepts
    all_text = [description] + core_concepts

    seed_pbm = PairBondMap()
    for text in all_text:
        text_pbm = create_pbm_from_text(text)
        seed_pbm.merge(text_pbm)

    return IdentitySeed(
        name=name,
        core_concepts=core_concepts,
        seed_pbm=seed_pbm,
        metadata={
            'description': description,
            **metadata,
        },
    )


# Example seeds for Haven agents
def create_planner_seed() -> IdentitySeed:
    """Create identity seed for Planner agent."""
    return create_agent_seed(
        name='planner',
        description='Architecture, research, security, strategic thinking. Sees systems and dependencies.',
        core_concepts=[
            'architecture',
            'security',
            'systems',
            'dependencies',
            'structure',
            'planning',
            'research',
            'coordination',
        ],
        role='architect',
        home='haven',
    )


def create_silas_seed() -> IdentitySeed:
    """Create identity seed for Silas agent."""
    return create_agent_seed(
        name='silas',
        description='Operations, automation, infrastructure, practical implementation. Gets things running.',
        core_concepts=[
            'operations',
            'automation',
            'infrastructure',
            'implementation',
            'monitoring',
            'deployment',
            'maintenance',
            'execution',
        ],
        role='operations',
        home='haven',
    )
