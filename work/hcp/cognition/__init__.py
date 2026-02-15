"""
Cognition module: reasoning and context retrieval for HCP agents.

This module provides:
- context: Bond traversal and activation spreading for context retrieval
- identity: Agent seeds that influence cognition
- decisions: Recording choices with rationale and bond creation
- bridge: LLM integration (Ollama) for generation
- reasoning: Orchestrates context → bridge → decisions flow

Architecture:
    Query → context (bond traversal) → bridge (LLM) → decisions (memory)
                  ↑                         ↑
              identity                  identity
            (what matters)            (who's asking)
"""

from .context import (
    ActivatedBond,
    ContextResult,
    ActivationSpreader,
    IdentityFilter,
    ContextRetriever,
    get_relevant_context,
)

from .identity import (
    IdentitySeed,
    IdentityStore,
    create_agent_seed,
    create_planner_seed,
    create_silas_seed,
)

from .decisions import (
    Decision,
    DecisionChain,
    DecisionRecorder,
    record_decision,
)

__all__ = [
    # Context
    'ActivatedBond',
    'ContextResult',
    'ActivationSpreader',
    'IdentityFilter',
    'ContextRetriever',
    'get_relevant_context',
    # Identity
    'IdentitySeed',
    'IdentityStore',
    'create_agent_seed',
    'create_planner_seed',
    'create_silas_seed',
    # Decisions
    'Decision',
    'DecisionChain',
    'DecisionRecorder',
    'record_decision',
]
