"""
LLM Bridge: Connect cognition to language model (Ollama).

This module handles the generation side of cognition:
- Connect to Ollama for LLM generation
- Format context bonds into prompts
- Parse responses back to tokens
- Stream responses with callbacks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Any
import json
import urllib.request
import urllib.error


@dataclass
class Bond:
    """Simple bond representation for context injection."""
    left: str
    right: str
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Format bond for inclusion in prompt."""
        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            return f"{self.left} -> {self.right} ({meta_str})"
        return f"{self.left} -> {self.right}"


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    model: str = "tinyllama:latest"
    temperature: float = 0.7
    max_tokens: int = 256
    system_prompt: str | None = None
    ollama_host: str = "http://localhost:11434"


@dataclass
class GenerationResult:
    """Result of LLM generation."""
    response: str
    model: str
    tokens_generated: int
    duration_ms: float
    context_bonds_used: int


class OllamaBridge:
    """
    Bridge to Ollama LLM for generation.

    Handles:
    - Health checks
    - Model listing
    - Generation with context
    - Streaming responses
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        """
        Initialize bridge.

        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()

    def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            url = f"{self.config.ollama_host}/api/tags"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            url = f"{self.config.ollama_host}/api/tags"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return [m['name'] for m in data.get('models', [])]
        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            return []

    def generate(
        self,
        query: str,
        context_bonds: list[Bond] | None = None,
        identity_token: str | None = None,
    ) -> GenerationResult:
        """
        Generate response using LLM.

        Args:
            query: The user query/prompt
            context_bonds: Relevant bonds for context
            identity_token: Identity string for agent persona

        Returns:
            GenerationResult with response and metadata
        """
        # Build prompt with context
        prompt = self._build_prompt(query, context_bonds, identity_token)

        # Make request
        url = f"{self.config.ollama_host}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if self.config.system_prompt:
            payload["system"] = self.config.system_prompt

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode('utf-8'))

                return GenerationResult(
                    response=data.get('response', ''),
                    model=data.get('model', self.config.model),
                    tokens_generated=data.get('eval_count', 0),
                    duration_ms=data.get('total_duration', 0) / 1_000_000,  # ns to ms
                    context_bonds_used=len(context_bonds) if context_bonds else 0,
                )

        except (urllib.error.URLError, json.JSONDecodeError) as e:
            return GenerationResult(
                response=f"Error: {str(e)}",
                model=self.config.model,
                tokens_generated=0,
                duration_ms=0,
                context_bonds_used=0,
            )

    def generate_stream(
        self,
        query: str,
        context_bonds: list[Bond] | None = None,
        identity_token: str | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        """
        Generate response with streaming.

        Args:
            query: The user query/prompt
            context_bonds: Relevant bonds for context
            identity_token: Identity string for agent persona
            callback: Function to call with each token

        Returns:
            GenerationResult with full response
        """
        prompt = self._build_prompt(query, context_bonds, identity_token)

        url = f"{self.config.ollama_host}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if self.config.system_prompt:
            payload["system"] = self.config.system_prompt

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            full_response = []
            tokens = 0
            duration = 0

            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        token = data.get('response', '')
                        if token:
                            full_response.append(token)
                            if callback:
                                callback(token)
                        if data.get('done'):
                            tokens = data.get('eval_count', 0)
                            duration = data.get('total_duration', 0) / 1_000_000

            return GenerationResult(
                response=''.join(full_response),
                model=self.config.model,
                tokens_generated=tokens,
                duration_ms=duration,
                context_bonds_used=len(context_bonds) if context_bonds else 0,
            )

        except (urllib.error.URLError, json.JSONDecodeError) as e:
            return GenerationResult(
                response=f"Error: {str(e)}",
                model=self.config.model,
                tokens_generated=0,
                duration_ms=0,
                context_bonds_used=0,
            )

    def _build_prompt(
        self,
        query: str,
        context_bonds: list[Bond] | None,
        identity_token: str | None,
    ) -> str:
        """Build prompt with context and identity."""
        parts = []

        # Add identity if provided
        if identity_token:
            parts.append(f"[Identity: {identity_token}]")

        # Add context bonds if provided
        if context_bonds:
            parts.append("[Context bonds:]")
            for bond in context_bonds[:10]:  # Limit to 10 bonds
                parts.append(f"  {bond.to_prompt_text()}")
            parts.append("")  # Blank line

        # Add query
        parts.append(query)

        return "\n".join(parts)


def generate(
    query: str,
    context_bonds: list[Bond] | None = None,
    identity_token: str | None = None,
    model: str = "tinyllama:latest",
) -> str:
    """
    Quick interface for generation.

    Args:
        query: What to ask the LLM
        context_bonds: Relevant context
        identity_token: Agent identity
        model: Which model to use

    Returns:
        Generated response string
    """
    bridge = OllamaBridge(GenerationConfig(model=model))
    result = bridge.generate(query, context_bonds, identity_token)
    return result.response


# CLI for testing
if __name__ == "__main__":
    import sys

    config = GenerationConfig(model="tinyllama:latest")
    bridge = OllamaBridge(config)

    print("Ollama Bridge CLI")
    print("-" * 40)

    if not bridge.health_check():
        print("Error: Ollama not available")
        sys.exit(1)

    print(f"Available models: {bridge.list_models()}")

    # Test with sample context
    context = [
        Bond("name", "planner", metadata={"type": "identity"}),
        Bond("role", "architecture", metadata={"type": "function"}),
    ]

    query = "Describe yourself briefly."
    print(f"\nQuery: {query}")
    print(f"Context: {len(context)} bonds")
    print("\nGenerating...")

    result = bridge.generate(query, context, "dA.AA.AA.AB")
    print(f"\nResponse: {result.response}")
    print(f"Tokens: {result.tokens_generated}, Duration: {result.duration_ms:.0f}ms")
