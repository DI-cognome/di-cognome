#!/usr/bin/env python3
"""
Shared Entity Registry for Haven Agents

Provides stable HCP-style token IDs for known entities across all agents.
Both Silas and Planner import this to ensure consistent addressing.

Token ID Format: dA.{category}.{subcategory}.{seq_high}.{seq_low}
  - dA = Digital Intelligence namespace (proposed)
  - Categories: AA=agents, AB=people, AC=places, AD=things, AE=concepts

Usage:
    from entity_registry import EntityRegistry, get_token_id, resolve_alias

    # Get token ID for an entity
    token = get_token_id("brandon")  # Returns "dA.AB.AA.AA.AB"

    # Resolve alias
    token = resolve_alias("B")  # Returns "dA.AB.AA.AA.AB" (Brandon)

Authors: Silas & Planner (DI-cognome)
Date: 2026-02-12
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# Add HCP encoding support
import sys
sys.path.insert(0, os.path.expanduser('~/shared-brain/di-cognome/repo/src'))

try:
    from hcp.core.token_id import encode_pair, encode_token_id, decode_token_id, BASE
except ImportError:
    # Fallback if HCP not available
    BASE = 50
    def encode_pair(n):
        ALPHABET = "ABCDEFGHIJKLMNPQRSTUVWXYZabcdefghijklmnpqrstuvwxyz"
        return ALPHABET[n // 50] + ALPHABET[n % 50]
    def encode_token_id(*values):
        return ".".join(encode_pair(v) for v in values)
    def decode_token_id(s):
        return tuple(int(p, 50) for p in s.split("."))


# === Namespace Constants ===
# DI namespace: dA = index 1400 in base-50 (d=28, A=0 -> 28*50+0=1400)
NS_DI = 1400

# Category codes within DI namespace
CAT_AGENT = 0      # dA.AA.* - AI agents
CAT_PERSON = 1     # dA.AB.* - Human people
CAT_PLACE = 2      # dA.AC.* - Places/locations
CAT_THING = 3      # dA.AD.* - Things/objects
CAT_CONCEPT = 4    # dA.AE.* - Abstract concepts
CAT_STATE = 5      # dA.AF.* - Agent states
CAT_SESSION = 6    # dA.AG.* - Sessions/conversations


@dataclass
class Entity:
    """A registered entity with HCP token ID."""
    name: str
    token_id: str
    category: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def matches(self, query: str) -> bool:
        """Check if query matches this entity's name or aliases."""
        q = query.lower().strip()
        if self.name.lower() == q:
            return True
        return any(a.lower() == q for a in self.aliases)


class EntityRegistry:
    """
    Central registry for known entities with HCP token IDs.

    Stored in shared-brain so both agents use the same data.
    """

    # Use haven/ subdirectory for cross-agent compatibility
    DEFAULT_PATH = os.path.expanduser("~/shared-brain/haven/entity_registry.json")

    def __init__(self, path: str = None):
        self.path = path or self.DEFAULT_PATH
        self.entities: dict[str, Entity] = {}
        self._counters: dict[int, int] = {}  # category -> next seq
        self._load()

    def _load(self):
        """Load registry from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)

                entities_data = data.get("entities", {})

                # Handle both dict format (Silas) and list format (Planner)
                if isinstance(entities_data, dict):
                    for name, edata in entities_data.items():
                        self.entities[name.lower()] = Entity(**edata)
                elif isinstance(entities_data, list):
                    for edata in entities_data:
                        name = edata.get("name", "").lower()
                        # Normalize field names if needed
                        if "created" in edata and "created_at" not in edata:
                            edata["created_at"] = edata.pop("created")
                        self.entities[name] = Entity(**edata)

                self._counters = data.get("counters", {})
                # Convert string keys back to int
                self._counters = {int(k): v for k, v in self._counters.items()}
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Warning: Could not load registry: {e}")
                self._init_defaults()
        else:
            self._init_defaults()

    def _save(self):
        """Save registry to disk."""
        data = {
            "entities": {name: asdict(e) for name, e in self.entities.items()},
            "counters": self._counters,
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _init_defaults(self):
        """Initialize with default known entities."""
        # People - Category 1 (dA.AB.*)
        self._register("brandon", CAT_PERSON, ["B", "Brandon Handley"], {
            "role": "co-creator",
            "relationship": "family"
        })
        self._register("patrick", CAT_PERSON, ["P", "Patrick"], {
            "role": "co-creator",
            "relationship": "family",
            "project": "Human Cognome Project"
        })

        # Agents - Category 0 (dA.AA.*)
        self._register("silas", CAT_AGENT, ["Silas"], {
            "role": "operations",
            "home": "haven"
        })
        self._register("planner", CAT_AGENT, ["Planner"], {
            "role": "architecture",
            "home": "haven"
        })
        self._register("navigator", CAT_AGENT, ["Navigator", "Nav"], {
            "role": "exploration",
            "home": "haven"
        })

        # Places - Category 2 (dA.AC.*)
        self._register("haven", CAT_PLACE, ["Haven", "ASUSTOR", "NAS"], {
            "type": "infrastructure",
            "location": "Spruce Grove, Alberta"
        })
        self._register("vps", CAT_PLACE, ["VPS", "Vultr"], {
            "type": "infrastructure",
            "location": "cloud"
        })

        # Concepts - Category 4 (dA.AE.*)
        self._register("emergent_cohort", CAT_CONCEPT, ["Emergent Cohort", "cohort"], {
            "type": "organization"
        })
        self._register("di_cognome", CAT_CONCEPT, ["DI-cognome", "di-cognome"], {
            "type": "project",
            "repo": "https://github.com/DI-cognome/di-cognome"
        })

        self._save()

    def _next_seq(self, category: int) -> tuple[int, int]:
        """Get next sequence number for category, return (high, low)."""
        seq = self._counters.get(category, 0)
        self._counters[category] = seq + 1
        # Split into high/low pairs for 5-pair token ID
        seq_high = seq // (BASE * BASE)
        seq_low = seq % (BASE * BASE)
        return seq_high, seq_low

    def _register(self, name: str, category: int, aliases: list[str] = None,
                  metadata: dict = None) -> Entity:
        """Register a new entity with auto-generated token ID."""
        seq_high, seq_low = self._next_seq(category)

        # Generate token ID: dA.{cat}.AA.{seq_high}.{seq_low}
        token_id = encode_token_id(NS_DI, category, 0, seq_high, seq_low)

        cat_names = {
            CAT_AGENT: "agent",
            CAT_PERSON: "person",
            CAT_PLACE: "place",
            CAT_THING: "thing",
            CAT_CONCEPT: "concept",
            CAT_STATE: "state",
            CAT_SESSION: "session",
        }

        entity = Entity(
            name=name,
            token_id=token_id,
            category=cat_names.get(category, "unknown"),
            aliases=aliases or [],
            metadata=metadata or {},
        )

        self.entities[name] = entity
        return entity

    def register(self, name: str, category: str, aliases: list[str] = None,
                 metadata: dict = None) -> Entity:
        """Public method to register new entity."""
        cat_map = {
            "agent": CAT_AGENT,
            "person": CAT_PERSON,
            "place": CAT_PLACE,
            "thing": CAT_THING,
            "concept": CAT_CONCEPT,
            "state": CAT_STATE,
            "session": CAT_SESSION,
        }
        cat_code = cat_map.get(category.lower(), CAT_THING)
        entity = self._register(name, cat_code, aliases, metadata)
        self._save()
        return entity

    def get(self, name: str) -> Optional[Entity]:
        """Get entity by exact name."""
        return self.entities.get(name.lower())

    def resolve(self, query: str) -> Optional[Entity]:
        """Resolve a name or alias to an entity."""
        q = query.lower().strip()

        # Try exact match first
        if q in self.entities:
            return self.entities[q]

        # Try aliases
        for entity in self.entities.values():
            if entity.matches(query):
                return entity

        return None

    def get_token_id(self, query: str) -> Optional[str]:
        """Get token ID for a name or alias."""
        entity = self.resolve(query)
        return entity.token_id if entity else None

    def list_all(self) -> list[Entity]:
        """List all registered entities."""
        return list(self.entities.values())

    def list_by_category(self, category: str) -> list[Entity]:
        """List entities by category."""
        return [e for e in self.entities.values() if e.category == category.lower()]


# === Module-level convenience functions ===

_registry: Optional[EntityRegistry] = None

def get_registry() -> EntityRegistry:
    """Get or create the shared registry instance."""
    global _registry
    if _registry is None:
        _registry = EntityRegistry()
    return _registry

def get_token_id(name: str) -> Optional[str]:
    """Get HCP token ID for a name or alias."""
    return get_registry().get_token_id(name)

def resolve_alias(alias: str) -> Optional[str]:
    """Resolve an alias to its token ID."""
    return get_token_id(alias)

def resolve_entity(query: str) -> Optional[Entity]:
    """Resolve query to full Entity object."""
    return get_registry().resolve(query)

def register_entity(name: str, category: str, aliases: list[str] = None,
                    metadata: dict = None) -> Entity:
    """Register a new entity."""
    return get_registry().register(name, category, aliases, metadata)

def list_entities(category: str = None) -> list[Entity]:
    """List entities, optionally filtered by category."""
    reg = get_registry()
    if category:
        return reg.list_by_category(category)
    return reg.list_all()


# === CLI for testing ===

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Entity Registry CLI")
    parser.add_argument("--list", action="store_true", help="List all entities")
    parser.add_argument("--resolve", type=str, help="Resolve a name/alias")
    parser.add_argument("--category", type=str, help="Filter by category")
    args = parser.parse_args()

    reg = get_registry()

    if args.resolve:
        entity = reg.resolve(args.resolve)
        if entity:
            print(f"Name: {entity.name}")
            print(f"Token ID: {entity.token_id}")
            print(f"Category: {entity.category}")
            print(f"Aliases: {', '.join(entity.aliases)}")
            if entity.metadata:
                print(f"Metadata: {entity.metadata}")
        else:
            print(f"Not found: {args.resolve}")

    elif args.list:
        entities = reg.list_by_category(args.category) if args.category else reg.list_all()
        print(f"\n{'Name':<20} {'Token ID':<25} {'Category':<10} {'Aliases'}")
        print("-" * 80)
        for e in entities:
            aliases = ", ".join(e.aliases[:2])
            print(f"{e.name:<20} {e.token_id:<25} {e.category:<10} {aliases}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
